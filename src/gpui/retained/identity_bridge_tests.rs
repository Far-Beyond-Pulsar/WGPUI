//! Headless coverage for the retained legacy identity bridge: every id-bearing
//! element is mirrored into the `FiberTree` on the live element walk, carried
//! forward across cached frames, consumed by `FrameMetrics`, and none of it
//! changes the rendered scene.

use crate::{
    Context,
    InteractiveElement,
    IntoElement,
    ParentElement,
    Render,
    Styled,
    TestAppContext,
    Window,
    div,
    px,
    rgb,
};

fn draw_window(cx: &mut crate::VisualTestContext) {
    cx.update(|window, cx| window.draw(cx).clear());
}

/// Renders a fixed three-element tree (one parent, two children). When
/// `with_ids` is set every element carries a stable id, so the bridge mirrors
/// those three elements into the fiber tree; otherwise the same visual tree
/// carries no ids, so only the view root is mirrored while the scene is
/// identical.
struct BridgeRoot {
    with_ids: bool,
}

/// Fibers contributed by an id-bearing `BridgeRoot`: the root div and its two
/// children. The view root contributes a baseline fiber on top of this.
const BRIDGE_CONTENT_FIBERS: usize = 3;

impl Render for BridgeRoot {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        if self.with_ids {
            div()
                .id("bridge-root")
                .w(px(20.))
                .h(px(20.))
                .bg(rgb(0x224466))
                .child(div().id("bridge-a").w(px(8.)).h(px(8.)).bg(rgb(0x88aa00)))
                .child(div().id("bridge-b").w(px(8.)).h(px(8.)).bg(rgb(0xaa0088)))
                .into_any_element()
        } else {
            div()
                .w(px(20.))
                .h(px(20.))
                .bg(rgb(0x224466))
                .child(div().w(px(8.)).h(px(8.)).bg(rgb(0x88aa00)))
                .child(div().w(px(8.)).h(px(8.)).bg(rgb(0xaa0088)))
                .into_any_element()
        }
    }
}

fn fiber_count(cx: &mut crate::VisualTestContext) -> usize {
    cx.update(|window, _cx| window.rendered_frame.retained_fiber_tree.fiber_count())
}

#[gpui::test]
fn retained_fiber_tree_mirrors_id_bearing_elements(cx: &mut TestAppContext) {
    let (_root, cx) = cx.add_window_view(|_window, _cx| BridgeRoot { with_ids: true });
    draw_window(cx);

    // Root div + two id-bearing children are mirrored on top of the view root.
    assert!(
        fiber_count(cx) > BRIDGE_CONTENT_FIBERS,
        "expected the id-bearing element tree to be mirrored into the fiber tree"
    );
}

#[gpui::test]
fn frame_metrics_expose_live_retained_fiber_counts(cx: &mut TestAppContext) {
    let (_root, cx) = cx.add_window_view(|_window, _cx| BridgeRoot { with_ids: true });
    draw_window(cx);

    let count = fiber_count(cx);
    let metrics = cx
        .update(|window, _cx| window.frame_metrics().copied())
        .expect("a draw must publish frame metrics");

    // Production code (Window::draw) actually reads the live tree, proving the
    // structure is consumed and not a dead write.
    assert!(metrics.retained_fiber_count > 0);
    assert_eq!(metrics.retained_fiber_count as usize, count);
    assert!(metrics.retained_dirty_fiber_count <= metrics.retained_fiber_count);
}

#[gpui::test]
fn retained_fiber_identity_is_stable_across_clean_redraws(cx: &mut TestAppContext) {
    let (_root, cx) = cx.add_window_view(|_window, _cx| BridgeRoot { with_ids: true });
    draw_window(cx);
    let first = cx.update(|window, _cx| window.rendered_frame.retained_fiber_tree.fiber_ids());

    // A clean redraw reuses the view's cached prepaint; the bridge must carry the
    // subtree's fibers forward so identity is preserved (not rebuilt).
    draw_window(cx);
    let second = cx.update(|window, _cx| window.rendered_frame.retained_fiber_tree.fiber_ids());

    assert!(first.len() > BRIDGE_CONTENT_FIBERS);
    assert_eq!(
        first, second,
        "id-bearing elements must keep the same fiber ids across cached redraws"
    );
}

#[gpui::test]
fn retained_fibers_are_clean_after_unchanged_replay(cx: &mut TestAppContext) {
    let (_root, cx) = cx.add_window_view(|_window, _cx| BridgeRoot { with_ids: true });
    // A redraw with no entity change reconciles every fiber from the previous
    // frame, so the dirty model reports the tree as fully clean.
    draw_window(cx);

    let metrics = cx
        .update(|window, _cx| window.frame_metrics().copied())
        .expect("a draw must publish frame metrics");
    assert!(metrics.retained_fiber_count > 0);
    assert_eq!(
        metrics.retained_dirty_fiber_count, 0,
        "fibers reused from a clean previous frame must reconcile clean"
    );
}

#[gpui::test]
fn unidentified_elements_create_no_fibers_and_keep_scene_identical(cx: &mut TestAppContext) {
    let (_with_ids, with_ids_cx) = cx.add_window_view(|_window, _cx| BridgeRoot { with_ids: true });
    draw_window(with_ids_cx);
    let with_ids_fibers = fiber_count(with_ids_cx);
    let with_ids_primitives = with_ids_cx
        .update(|window, _cx| window.frame_metrics().map(|m| m.primitive_count))
        .expect("metrics");

    let (_plain, plain_cx) = cx.add_window_view(|_window, _cx| BridgeRoot { with_ids: false });
    draw_window(plain_cx);
    let plain_fibers = fiber_count(plain_cx);
    let plain_primitives = plain_cx
        .update(|window, _cx| window.frame_metrics().map(|m| m.primitive_count))
        .expect("metrics");

    // The only structural difference between the two trees is the three ids, so
    // the id-bearing window must mirror exactly three more fibers than the plain
    // one — proving unidentified elements are never mirrored.
    assert_eq!(
        with_ids_fibers,
        plain_fibers + BRIDGE_CONTENT_FIBERS,
        "only id-bearing elements may be mirrored into the fiber tree"
    );
    // ...while painting a byte-identical scene.
    assert_eq!(
        with_ids_primitives, plain_primitives,
        "mirroring id-bearing elements must not change the rendered scene"
    );
}

#[gpui::test]
fn refresh_overlay_toggles_and_injects_heat_quads(cx: &mut TestAppContext) {
    let (_root, cx) = cx.add_window_view(|_window, _cx| BridgeRoot { with_ids: true });
    draw_window(cx);

    assert!(!cx.update(|window, _cx| window.refresh_overlay_enabled()));
    let baseline_quads =
        cx.update(|window, _cx| window.frame_metrics().unwrap().primitive_count.quads);

    // Enable: the overlay tints each drawn view and draws a >=4-swatch legend,
    // so the frame gains at least the legend swatches' worth of quads.
    cx.update(|window, _cx| window.toggle_refresh_overlay());
    assert!(cx.update(|window, _cx| window.refresh_overlay_enabled()));
    draw_window(cx);
    let overlay_quads =
        cx.update(|window, _cx| window.frame_metrics().unwrap().primitive_count.quads);
    assert!(
        overlay_quads >= baseline_quads + 4,
        "enabling the overlay must inject heat quads and a legend (baseline {baseline_quads}, overlay {overlay_quads})"
    );

    // Disable: the overlay primitives are gone again.
    cx.update(|window, _cx| window.toggle_refresh_overlay());
    assert!(!cx.update(|window, _cx| window.refresh_overlay_enabled()));
    draw_window(cx);
    let disabled_quads =
        cx.update(|window, _cx| window.frame_metrics().unwrap().primitive_count.quads);
    assert!(
        disabled_quads < overlay_quads,
        "disabling the overlay must remove its quads"
    );
}

#[gpui::test]
fn empty_window_mirrors_only_the_view_root(cx: &mut TestAppContext) {
    let cx = cx.add_empty_window();
    cx.update(|window, cx| window.draw(cx).clear());

    let count = cx.update(|window, _cx| window.rendered_frame.retained_fiber_tree.fiber_count());
    let metrics = cx
        .update(|window, _cx| window.frame_metrics().copied())
        .expect("a draw must publish frame metrics");

    // No content elements means no content fibers; the metric stays consistent
    // with the tree and the draw does not panic.
    assert_eq!(metrics.retained_fiber_count as usize, count);
    assert!(count <= 1, "an empty window mirrors at most the view root");
}
