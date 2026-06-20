//! Interactive-path and scene-introspection audit tests.
//!
//! These run headlessly on the test platform (no GPU/pixels) and assert on
//! observable state after driving real interactions: window resize, focus,
//! window activation, committed typing, IME marked/preedit text, list
//! scroll/splice/reset, and that views actually emit the expected scene
//! primitives (built-in quads and custom render-primitive batches).
//!
//! Pixel-level correctness of the GPU output is covered separately by the
//! real-GPU offscreen harness (`platform::cross::audit`) and the on-display
//! `render_audit` example.

use std::ops::Range;

use gpui_prim_solid::{SolidQuad, SolidQuads};

use crate::{
    App,
    Bounds,
    Context,
    ElementInputHandler,
    EntityInputHandler,
    FocusHandle,
    Focusable,
    InteractiveElement,
    IntoElement,
    ListAlignment,
    ListState,
    ParentElement,
    Pixels,
    Point,
    Render,
    ScrollStrategy,
    Styled,
    TestAppContext,
    UTF16Selection,
    UniformListScrollHandle,
    Window,
    canvas,
    div,
    px,
    rgb,
    size,
    uniform_list,
};

// ---- test views -----------------------------------------------------------

/// A minimal focusable, full-window view used for resize/focus/activation tests.
struct FocusView {
    focus: FocusHandle,
}

impl Focusable for FocusView {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl Render for FocusView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("focus-root")
            .track_focus(&self.focus)
            .size_full()
            .bg(rgb(0x101418))
    }
}

/// A plain view that paints a solid background, so the scene has at least one
/// built-in quad to assert on.
struct BackgroundView;

impl Render for BackgroundView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div().size_full().bg(rgb(0x223344))
    }
}

/// Paints `count` custom (plugin) `SolidQuad` primitives via a canvas, so the
/// scene's custom-primitive batches can be asserted on.
struct CustomPrimitiveView {
    count: usize,
}

impl Render for CustomPrimitiveView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let count = self.count;
        div().w(px(200.)).h(px(200.)).child(
            canvas(
                |_bounds, _window, _cx| (),
                move |bounds, _, window, _cx| {
                    for index in 0..count {
                        let quad = SolidQuad::new(
                            [index as f32 * 4.0, 0.0],
                            [3.0, 3.0],
                            [0.2, 0.6, 1.0, 1.0],
                        );
                        window.paint_custom_primitive(SolidQuads::TYPE, quad.as_bytes(), bounds);
                    }
                },
            )
            .size_full(),
        )
    }
}

/// A uniform-list view whose scroll is driven by an external handle. It records
/// the visible item range each render, which is how the codebase's own tests
/// observe scrolling.
struct UniformListView {
    scroll: UniformListScrollHandle,
    visible_range: Range<usize>,
}

impl Render for UniformListView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div().size_full().child(
            uniform_list(
                "rows",
                200,
                cx.processor(|this, range: Range<usize>, _window, _cx| {
                    this.visible_range = range.clone();
                    range
                        .map(|index| div().id(index).h(px(20.)).child(format!("row {index}")))
                        .collect::<Vec<_>>()
                }),
            )
            .track_scroll(&self.scroll)
            .h(px(200.)),
        )
    }
}

/// A focusable text input that installs an [`ElementInputHandler`] each paint,
/// so committed typing and IME marked text can be driven against it.
struct TextInputView {
    focus: FocusHandle,
    text: String,
    marked: Option<Range<usize>>,
}

impl TextInputView {
    fn target_range(&self, range: Option<Range<usize>>) -> Range<usize> {
        let len = self.text.len();
        let range = range.or_else(|| self.marked.clone()).unwrap_or(len..len);
        range.start.min(len)..range.end.min(len)
    }
}

impl Focusable for TextInputView {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl Render for TextInputView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let view = cx.entity();
        let focus = self.focus.clone();
        div().track_focus(&self.focus).size_full().child(
            canvas(
                |_bounds, _window, _cx| (),
                move |bounds, _, window, cx| {
                    window.handle_input(&focus, ElementInputHandler::new(bounds, view.clone()), cx);
                },
            )
            .size_full(),
        )
    }
}

impl EntityInputHandler for TextInputView {
    fn text_for_range(
        &mut self,
        range: Range<usize>,
        adjusted_range: &mut Option<Range<usize>>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<String> {
        let len = self.text.len();
        let clamped = range.start.min(len)..range.end.min(len);
        *adjusted_range = Some(clamped.clone());
        self.text.get(clamped).map(|slice| slice.to_string())
    }

    fn selected_text_range(
        &mut self,
        _ignore_disabled_input: bool,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<UTF16Selection> {
        let end = self.text.len();
        Some(UTF16Selection {
            range: end..end,
            reversed: false,
        })
    }

    fn marked_text_range(
        &self,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<Range<usize>> {
        self.marked.clone()
    }

    fn unmark_text(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        self.marked = None;
        cx.notify();
    }

    fn replace_text_in_range(
        &mut self,
        range: Option<Range<usize>>,
        text: &str,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let target = self.target_range(range);
        self.text.replace_range(target, text);
        self.marked = None;
        cx.notify();
    }

    fn replace_and_mark_text_in_range(
        &mut self,
        range: Option<Range<usize>>,
        new_text: &str,
        _new_selected_range: Option<Range<usize>>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let target = self.target_range(range);
        let start = target.start;
        self.text.replace_range(target, new_text);
        self.marked = Some(start..start + new_text.len());
        cx.notify();
    }

    fn bounds_for_range(
        &mut self,
        _range_utf16: Range<usize>,
        element_bounds: Bounds<Pixels>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<Bounds<Pixels>> {
        Some(element_bounds)
    }

    fn character_index_for_point(
        &mut self,
        _point: Point<Pixels>,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> Option<usize> {
        Some(0)
    }
}

// ---- interactive-path tests ----------------------------------------------

#[gpui::test]
fn resize_updates_viewport_size(cx: &mut TestAppContext) {
    let (_view, cx) = cx.add_window_view(|_window, cx| FocusView {
        focus: cx.focus_handle(),
    });

    cx.simulate_resize(size(px(640.), px(480.)));
    cx.run_until_parked();

    let viewport = cx.update(|window, _cx| window.viewport_size());
    assert_eq!(viewport, size(px(640.), px(480.)));
}

#[gpui::test]
fn focus_is_reflected_in_window(cx: &mut TestAppContext) {
    let (view, cx) = cx.add_window_view(|_window, cx| FocusView {
        focus: cx.focus_handle(),
    });
    let handle = cx.update(|_window, cx| view.read(cx).focus.clone());

    cx.update(|window, cx| window.focus(&handle, cx));
    cx.run_until_parked();
    assert!(cx.update(|window, _cx| handle.is_focused(window)));

    cx.update(|window, _cx| window.blur());
    cx.run_until_parked();
    assert!(!cx.update(|window, _cx| handle.is_focused(window)));
}

#[gpui::test]
fn window_activation_toggles_active_state(cx: &mut TestAppContext) {
    let (_view, cx) = cx.add_window_view(|_window, cx| FocusView {
        focus: cx.focus_handle(),
    });

    cx.deactivate_window();
    assert!(!cx.update(|window, _cx| window.is_window_active()));

    cx.activate_window();
    assert!(cx.update(|window, _cx| window.is_window_active()));
}

#[gpui::test]
fn committed_typing_updates_text(cx: &mut TestAppContext) {
    let (input, cx) = cx.add_window_view(|_window, cx| TextInputView {
        focus: cx.focus_handle(),
        text: String::new(),
        marked: None,
    });

    let handle = cx.update(|_window, cx| input.read(cx).focus.clone());
    cx.update(|window, cx| window.focus(&handle, cx));
    cx.run_until_parked();
    // Paint so the focused element installs its input handler this frame.
    cx.update(|window, cx| {
        window.draw(cx).clear();
    });

    cx.simulate_input("hi");

    let text = cx.update(|_window, cx| input.read(cx).text.clone());
    assert_eq!(text, "hi");
}

#[gpui::test]
fn ime_marked_text_drives_preedit(cx: &mut TestAppContext) {
    let (input, cx) = cx.add_window_view(|_window, cx| TextInputView {
        focus: cx.focus_handle(),
        text: String::new(),
        marked: None,
    });

    let handle = cx.update(|_window, cx| input.read(cx).focus.clone());
    cx.update(|window, cx| window.focus(&handle, cx));
    cx.run_until_parked();
    cx.update(|window, cx| {
        window.draw(cx).clear();
    });

    // Begin a composition (marked/preedit text) — the path ordinary keystroke
    // simulation never reaches.
    cx.simulate_ime_marked_text(None, "ab", Some(0..2));
    let (text, marked) =
        cx.update(|_window, cx| (input.read(cx).text.clone(), input.read(cx).marked.clone()));
    assert_eq!(text, "ab");
    assert_eq!(marked, Some(0..2));

    // Committing replaces the marked range and clears the composition.
    cx.simulate_input("X");
    let (text, marked) =
        cx.update(|_window, cx| (input.read(cx).text.clone(), input.read(cx).marked.clone()));
    assert_eq!(text, "X");
    assert_eq!(marked, None);
}

// ---- scene-introspection tests -------------------------------------------

#[gpui::test]
fn background_view_emits_quads(cx: &mut TestAppContext) {
    let (_view, cx) = cx.add_window_view(|_window, _cx| BackgroundView);
    cx.update(|window, cx| {
        window.refresh();
        window.draw(cx).clear();
    });

    let quad_count = cx.update(|window, _cx| window.rendered_frame.scene.quads.len());
    assert!(
        quad_count > 0,
        "a full-window background div should emit at least one quad, got {quad_count}"
    );
}

#[gpui::test]
fn custom_primitives_appear_in_scene_batches(cx: &mut TestAppContext) {
    let (_view, cx) = cx.add_window_view(|_window, _cx| CustomPrimitiveView { count: 5 });
    cx.update(|window, cx| {
        window.refresh();
        window.draw(cx).clear();
    });

    let (type_count, metric_count) = cx.update(|window, _cx| {
        (
            window.rendered_frame.scene.custom_primitives.type_count(),
            window
                .frame_metrics()
                .map(|metrics| metrics.custom_primitive_count)
                .unwrap_or(0),
        )
    });
    assert_eq!(type_count, 1, "one custom-primitive type was painted");
    assert_eq!(metric_count, 5, "five instances were collected");
}

// ---- list tests -----------------------------------------------------------

#[gpui::test]
fn uniform_list_scroll_advances_top_index(cx: &mut TestAppContext) {
    let scroll = UniformListScrollHandle::new();
    let (view, cx) = cx.add_window_view({
        let scroll = scroll.clone();
        move |_window, _cx| UniformListView {
            scroll,
            visible_range: 0..0,
        }
    });

    cx.update(|window, cx| {
        window.refresh();
        window.draw(cx).clear();
    });
    // Initially the top of the list is visible.
    let initial = view.read_with(cx, |view, _cx| view.visible_range.clone());
    assert_eq!(
        initial.start, 0,
        "list should start at the top, got {initial:?}"
    );

    scroll.scroll_to_item(120, ScrollStrategy::Top);
    cx.update(|window, cx| {
        window.refresh();
        window.draw(cx).clear();
    });

    let scrolled = view.read_with(cx, |view, _cx| view.visible_range.clone());
    assert!(
        scrolled.start > 0 && scrolled.contains(&120),
        "scrolling to item 120 should move the visible window to include it, got {scrolled:?}"
    );
}

#[test]
fn list_state_splice_append_and_reset() {
    // ListState bookkeeping is exercisable without a window.
    let state = ListState::new(0, ListAlignment::Top, px(0.));
    state.splice(0..0, 100); // append 100 items
    state.splice(10..20, 5); // splice a range
    state.reset(50); // reset to a new count

    // Reset returns scrolling to the top.
    let top = state.logical_scroll_top();
    assert_eq!(top.item_ix, 0);
}
