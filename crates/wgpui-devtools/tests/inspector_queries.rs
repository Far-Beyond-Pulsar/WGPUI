#![cfg(feature = "inspector")]

use wgpui_devtools::{
    BoundaryId, ElementId, ElementQuery, ElementRecord, Inspector, InstanceKey, Rect, ScrollRootId,
    SelectionError, SourceLocation, TileCoord,
};

fn bounds(x: f32, y: f32, width: f32, height: f32) -> Rect {
    Rect::from_origin_size([x, y], [width, height])
}

#[test]
fn public_inspector_queries_preserve_traversal_order_and_selection_is_diagnostic_only() {
    let first = ElementRecord::new(
        InstanceKey::from_raw(11),
        "first",
        bounds(0.0, 0.0, 8.0, 8.0),
    )
    .with_explicit_id("row")
    .with_source_location(SourceLocation::new("panel.rs", 21).with_column(4))
    .with_boundary(BoundaryId::from_raw(3))
    .with_scroll_root(ScrollRootId::from_raw(9))
    .with_tile(TileCoord::new(-1, 2));
    let second = ElementRecord::new(
        InstanceKey::from_raw(12),
        "second",
        bounds(8.0, 0.0, 8.0, 8.0),
    )
    .with_explicit_id(ElementId::from("row"));
    let mut inspector = Inspector::new();
    inspector.replace_records(vec![first, second]);

    let matches = inspector.query(&ElementQuery::ExplicitId(ElementId::from("row")));
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].address, InstanceKey::from_raw(11));
    assert_eq!(matches[1].address, InstanceKey::from_raw(12));
    assert_eq!(
        inspector
            .query_all(&ElementQuery::SourceLocation(SourceLocation::new(
                "panel.rs", 21
            )))
            .len(),
        1
    );
    assert_eq!(
        inspector
            .query_all(&ElementQuery::Bounds(bounds(7.0, 1.0, 2.0, 2.0)))
            .len(),
        2
    );

    assert_eq!(
        inspector.select(InstanceKey::from_raw(11)),
        Err(SelectionError::CaptureInactive)
    );
    assert!(inspector.arm_capture());
    assert!(inspector.begin_capture());
    let selected = inspector.select_query(&ElementQuery::StableAddress(InstanceKey::from_raw(11)));
    assert!(selected.is_ok());
    let selected = selected.unwrap_or_else(|_| unreachable!("record exists in the capture"));
    assert_eq!(selected.diagnostic_damage, Some(bounds(0.0, 0.0, 8.0, 8.0)));
    assert!(!selected.requires_layout);
    assert!(!selected.requires_scene_rebuild);
    assert_eq!(inspector.snapshot().elements.len(), 2);
}
