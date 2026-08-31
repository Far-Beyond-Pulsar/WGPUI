//! Creating Components Example
//!
//! This example demonstrates three different approaches to creating interactive
//! stateful components in GPUI:
//!
//! 1. `use_state` - Hook-like state scoped to an element's lifetime
//! 2. `RenderOnce` - Stateless component that receives state from parent
//! 3. `Render` - Entity-backed view with persistent internal state

#[path = "../prelude.rs"]
mod example_prelude;

use example_prelude::init_example;
use wgpui::{
    App, Application, Bounds, Colors, Context, Entity, IntoElement, Render, RenderOnce, WeakEntity,
    WindowBounds, WindowOptions, div, prelude::*, px, size,
};

// ============================================================================
// Approach 1: use_state
// ============================================================================
//
// `use_state` creates element-scoped state that persists across renders.
// It's similar to React's useState hook. The state is automatically tied
// to the element's identity via caller location or a provided key.
//
// Pros:
// - Simple, hook-like API
// - State is scoped to element lifetime
// - No boilerplate for simple state
//
// Cons:
// - Less explicit than Entity-backed state
// - State is tied to call site location

fn use_state_counter(
    colors: &Colors,
    parent: &WeakEntity<CreatingComponentsExample>,
    count: i32,
) -> impl IntoElement + 'static {
    let error = colors.error;
    let error_hover = colors.error_hover;
    let success = colors.success;
    let success_hover = colors.success_hover;

    div()
        .id("use-state-counter")
        .flex()
        .flex_col()
        .gap_2()
        .p_4()
        .rounded_lg()
        .bg(colors.surface)
        .child(
            div()
                .text_sm()
                .text_color(colors.text_muted)
                .child("use_state Counter"),
        )
        .child(
            div()
                .text_2xl()
                .text_color(colors.text)
                .child(format!("{}", count)),
        )
        .child(
            div()
                .flex()
                .gap_2()
                .child(
                    div()
                        .id("use-state-decrement")
                        .px_3()
                        .py_1()
                        .rounded_md()
                        .bg(error)
                        .text_color(colors.selected_text)
                        .cursor_pointer()
                        .hover(move |style| style.bg(error_hover))
                        .child("−")
                        .on_click({
                            let parent = parent.clone();
                            move |_, _, _cx| {
                                if let Err(error) = parent.update(|parent, cx| {
                                    parent.use_state_count -= 1;
                                    cx.notify();
                                }) {
                                    eprintln!("use-state counter update failed: {error}");
                                }
                            }
                        }),
                )
                .child(
                    div()
                        .id("use-state-increment")
                        .px_3()
                        .py_1()
                        .rounded_md()
                        .bg(success)
                        .text_color(colors.selected_text)
                        .cursor_pointer()
                        .hover(move |style| style.bg(success_hover))
                        .child("+")
                        .on_click({
                            let parent = parent.clone();
                            move |_, _, _| {
                                if let Err(error) = parent.update(|parent, cx| {
                                    parent.use_state_count += 1;
                                    cx.notify();
                                }) {
                                    eprintln!("use-state counter update failed: {error}");
                                }
                            }
                        }),
                ),
        )
}

// ============================================================================
// Approach 2: RenderOnce
// ============================================================================
//
// `RenderOnce` components are stateless and consumed when rendered.
// They receive all data as props and delegate state management to the parent.
// This is the recommended approach for presentational components.
//
// Pros:
// - Clear data flow (props down, events up)
// - Lightweight (no Entity allocation)
// - Easy to test
// - Highly composable
//
// Cons:
// - Cannot maintain internal state
// - Parent must manage all state

type AppCallback = Box<dyn Fn(&mut App) + 'static>;

#[derive(IntoElement)]
struct RenderOnceCounter {
    colors: Colors,
    count: i32,
    on_increment: Option<AppCallback>,
    on_decrement: Option<AppCallback>,
}

impl RenderOnceCounter {
    fn new(colors: Colors, count: i32) -> Self {
        Self {
            colors,
            count,
            on_increment: None,
            on_decrement: None,
        }
    }

    fn on_increment(mut self, callback: impl Fn(&mut App) + 'static) -> Self {
        self.on_increment = Some(Box::new(callback));
        self
    }

    fn on_decrement(mut self, callback: impl Fn(&mut App) + 'static) -> Self {
        self.on_decrement = Some(Box::new(callback));
        self
    }
}

impl RenderOnce for RenderOnceCounter {
    fn render(self) -> impl IntoElement + 'static {
        let colors = self.colors;
        let error = colors.error;
        let error_hover = colors.error_hover;
        let success = colors.success;
        let success_hover = colors.success_hover;

        div()
            .id("render-once-counter")
            .flex()
            .flex_col()
            .gap_2()
            .p_4()
            .rounded_lg()
            .bg(colors.surface)
            .child(
                div()
                    .text_sm()
                    .text_color(colors.text_muted)
                    .child("RenderOnce Counter"),
            )
            .child(
                div()
                    .text_2xl()
                    .text_color(colors.text)
                    .child(format!("{}", self.count)),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .child(
                        div()
                            .id("render-once-decrement")
                            .px_3()
                            .py_1()
                            .rounded_md()
                            .bg(error)
                            .text_color(colors.selected_text)
                            .cursor_pointer()
                            .hover(move |style| style.bg(error_hover))
                            .child("−")
                            .when_some(self.on_decrement, |element, callback| {
                                element.on_click(move |_, _, cx| callback(cx))
                            }),
                    )
                    .child(
                        div()
                            .id("render-once-increment")
                            .px_3()
                            .py_1()
                            .rounded_md()
                            .bg(success)
                            .text_color(colors.selected_text)
                            .cursor_pointer()
                            .hover(move |style| style.bg(success_hover))
                            .child("+")
                            .when_some(self.on_increment, |element, callback| {
                                element.on_click(move |_, _, cx| callback(cx))
                            }),
                    ),
            )
    }
}

// ============================================================================
// Approach 3: Render (Entity-backed)
// ============================================================================
//
// `Render` components are backed by an `Entity<T>` and maintain their own
// internal state. This is the recommended approach for complex components
// that need to manage their own state, subscribe to events, or spawn tasks.
//
// Pros:
// - Full control over internal state
// - Can subscribe to events and observe other entities
// - Can spawn async tasks
// - Has identity (can be passed around as Entity<T>)
//
// Cons:
// - More boilerplate
// - Higher memory overhead
// - More complex lifecycle

struct RenderCounter {
    count: i32,
    handle: WeakEntity<Self>,
}

impl RenderCounter {
    fn new(cx: &mut Context<Self>) -> Self {
        Self {
            count: 0,
            handle: cx.entity().downgrade(),
        }
    }

    fn increment(&mut self, cx: &mut Context<Self>) {
        self.count += 1;
        cx.notify();
    }

    fn decrement(&mut self, cx: &mut Context<Self>) {
        self.count -= 1;
        cx.notify();
    }
}

impl Render for RenderCounter {
    fn render(&mut self) -> impl IntoElement + 'static {
        let colors = Colors::for_appearance(&());
        let error = colors.error;
        let error_hover = colors.error_hover;
        let success = colors.success;
        let success_hover = colors.success_hover;

        div()
            .id("render-counter")
            .flex()
            .flex_col()
            .gap_2()
            .p_4()
            .rounded_lg()
            .bg(colors.surface)
            .child(
                div()
                    .text_sm()
                    .text_color(colors.text_muted)
                    .child("Render Counter"),
            )
            .child(
                div()
                    .text_2xl()
                    .text_color(colors.text)
                    .child(format!("{}", self.count)),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .child(
                        div()
                            .id("render-decrement")
                            .px_3()
                            .py_1()
                            .rounded_md()
                            .bg(error)
                            .text_color(colors.selected_text)
                            .cursor_pointer()
                            .hover(move |style| style.bg(error_hover))
                            .child("−")
                            .on_click({
                                let handle = self.handle.clone();
                                move |_, _, _| {
                                    if let Err(error) = handle.update(|this, cx| this.decrement(cx))
                                    {
                                        eprintln!("render counter update failed: {error}");
                                    }
                                }
                            }),
                    )
                    .child(
                        div()
                            .id("render-increment")
                            .px_3()
                            .py_1()
                            .rounded_md()
                            .bg(success)
                            .text_color(colors.selected_text)
                            .cursor_pointer()
                            .hover(move |style| style.bg(success_hover))
                            .child("+")
                            .on_click({
                                let handle = self.handle.clone();
                                move |_, _, _| {
                                    if let Err(error) = handle.update(|this, cx| this.increment(cx))
                                    {
                                        eprintln!("render counter update failed: {error}");
                                    }
                                }
                            }),
                    ),
            )
    }
}

// ============================================================================
// Main Application
// ============================================================================

struct CreatingComponentsExample {
    render_counter: Entity<RenderCounter>,
    render_once_count: i32,
    use_state_count: i32,
    handle: WeakEntity<Self>,
}

impl CreatingComponentsExample {
    fn new(cx: &mut Context<Self>) -> Self {
        Self {
            render_counter: cx.new(RenderCounter::new),
            render_once_count: 0,
            use_state_count: 0,
            handle: cx.entity().downgrade(),
        }
    }
}

impl Render for CreatingComponentsExample {
    fn render(&mut self) -> impl IntoElement + 'static {
        let colors = Colors::for_appearance(&());
        let render_once_count = self.render_once_count;
        let handle = self.handle.clone();

        div()
            .id("main")
            .size_full()
            .flex()
            .flex_col()
            .gap_6()
            .p_8()
            .bg(colors.background)
            .overflow_scroll()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(
                        div()
                            .text_2xl()
                            .font_weight(wgpui::FontWeight::BOLD)
                            .text_color(colors.text)
                            .child("Creating Components"),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(colors.text_muted)
                            .child("Three approaches to stateful components in GPUI"),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap_4()
                    .child(use_state_counter(
                        &colors,
                        &self.handle,
                        self.use_state_count,
                    ))
                    .child(
                        RenderOnceCounter::new(colors.clone(), render_once_count)
                            .on_increment({
                                let handle = handle.clone();
                                move |_cx| {
                                    handle
                                        .update(|this, cx| {
                                            this.render_once_count += 1;
                                            cx.notify();
                                        })
                                        .unwrap_or_else(|error| {
                                            eprintln!("render-once counter update failed: {error}");
                                        });
                                }
                            })
                            .on_decrement(move |_cx| {
                                handle
                                    .update(|this, cx| {
                                        this.render_once_count -= 1;
                                        cx.notify();
                                    })
                                    .unwrap_or_else(|error| {
                                        eprintln!("render-once counter update failed: {error}");
                                    });
                            }),
                    )
                    .child(self.render_counter.clone()),
            )
    }
}

fn main() {
    if let Err(error) = Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None::<()>, size(px(700.), px(400.)), cx);
        if let Err(error) = cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(CreatingComponentsExample::new),
        ) {
            eprintln!("failed to open creating-components window: {error}");
            cx.quit();
            return;
        }

        init_example(cx, "Creating Components");
    }) {
        eprintln!("native application failed: {error}");
    }
}
