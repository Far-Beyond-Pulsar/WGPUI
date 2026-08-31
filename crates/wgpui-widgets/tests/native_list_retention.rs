use wgpui_core::geometry::{Bounds, Point, Size};
use wgpui_core::invalidation::request::FrameSignals;
use wgpui_core::patch::apply::apply;
use wgpui_core::patch::emit::Emitter;
use wgpui_core::reconcile::plan::NodeOutcome;
use wgpui_core::reconcile::reconciler::Reconciler;
use wgpui_core::scene::Scene;
use wgpui_layout::taffy_tree::{LayoutTree, definite};
use wgpui_widgets::div::div;
use wgpui_widgets::list::uniform_list::uniform_list;
use wgpui_widgets::scroll::ScrollHandle;
use wgpui_widgets::styled::Styled;

fn frame(
    root: impl wgpui_core::element::IntoElement,
    reconciler: &mut Reconciler,
    layout: &mut LayoutTree,
    emitter: &mut Emitter,
    scene: &mut Scene,
) -> (
    wgpui_core::reconcile::FramePlan,
    wgpui_core::patch::emit::FrameEmission,
) {
    let plan = reconciler
        .reconcile(root.into_description(), layout)
        .expect("reconcile");
    let root = plan.root().expect("root").layout_node;
    layout
        .compute_layout(root, definite(220.0, 100.0))
        .expect("layout");
    let emission = emitter
        .emit(&plan, layout, &FrameSignals::new(), scene)
        .expect("emit");
    apply(scene, &emission.patch).expect("apply");
    (plan, emission)
}

#[test]
fn a_resident_uniform_list_scroll_is_transform_only_and_damages_its_viewport() {
    let handle = ScrollHandle::new();
    handle.set_viewport(
        Bounds::new(Point::default(), Size::pixels(100.0, 80.0)),
        Size::pixels(100.0, 1_000.0),
    );
    handle.set_offset(Point::new(
        wgpui_core::geometry::Pixels::ZERO,
        wgpui_core::geometry::Pixels(-0.25),
    ));

    let make_root = || {
        div()
            .w(220.0)
            .h(100.0)
            .child(
                uniform_list(100, Size::pixels(100.0, 10.0), |index| {
                    div().h(10.0).child(format!("row {index}"))
                })
                .w(100.0)
                .h(80.0)
                .overscan(2)
                .track_scroll(&handle),
            )
            .child(div().w(100.0).h(100.0).child("unrelated"))
    };

    let mut reconciler = Reconciler::new();
    let mut layout = LayoutTree::new();
    let mut emitter = Emitter::new();
    let mut scene = Scene::new();
    let _ = frame(
        make_root(),
        &mut reconciler,
        &mut layout,
        &mut emitter,
        &mut scene,
    );

    handle.scroll_by(Point::new(
        wgpui_core::geometry::Pixels::ZERO,
        wgpui_core::geometry::Pixels(-0.25),
    ));
    let (second, emission) = frame(
        make_root(),
        &mut reconciler,
        &mut layout,
        &mut emitter,
        &mut scene,
    );

    assert!(
        second
            .nodes()
            .iter()
            .all(|node| node.outcome == NodeOutcome::Reused)
    );
    assert_eq!(emission.patch.len(), 0);
    assert!(emission.stats.transform_only >= 1);
    assert_eq!(emission.damage.len(), 1);
    assert_eq!(emission.damage[0].width(), 100.0);
    assert_eq!(emission.damage[0].height(), 80.0);
    assert_eq!(second.stats().rebuilt, 0);
}
