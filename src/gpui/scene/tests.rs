use super::*;
use crate::{Bounds, ContentMask, EntityId, Point, ScaledPixels, Size};

fn test_bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
    Bounds {
        origin: Point {
            x: ScaledPixels(x),
            y: ScaledPixels(y),
        },
        size: Size {
            width: ScaledPixels(width),
            height: ScaledPixels(height),
        },
    }
}

fn test_quad(bounds: Bounds<ScaledPixels>) -> Quad {
    Quad {
        bounds,
        content_mask: ContentMask { bounds },
        ..Default::default()
    }
}

#[test]
fn generation_advances_for_dirty_chunks_not_clean_reuse() {
    let view_id = EntityId::from(1);
    let mut scene = Scene::default();

    scene.begin_view_chunk(view_id);
    scene.mark_chunk_clean(view_id);
    scene.end_view_chunk();
    scene.finish();

    assert_eq!(scene.scene_generation(), 0);

    scene.begin_view_chunk(view_id);
    scene.end_view_chunk();
    scene.finish();

    assert_eq!(scene.scene_generation(), 1);
}

#[test]
fn changed_ranges_survive_finish_after_dirty_chunks_are_cleared() {
    let view_id = EntityId::from(1);
    let mut scene = Scene::default();

    scene.begin_view_chunk(view_id);
    scene.insert_primitive(test_quad(test_bounds(0.0, 0.0, 10.0, 10.0)));
    scene.end_view_chunk();
    scene.finish();

    assert_eq!(scene.dirty_chunk_count(), 0);
    assert_eq!(scene.changed_ranges().quads, vec![0..1]);
}

#[test]
fn damage_includes_previous_bounds_for_dirty_view_chunks() {
    let view_id = EntityId::from(1);
    let mut previous_scene = Scene::default();

    previous_scene.begin_view_chunk(view_id);
    previous_scene.insert_primitive(test_quad(test_bounds(0.0, 0.0, 10.0, 10.0)));
    previous_scene.end_view_chunk();
    previous_scene.finish();

    let mut scene = Scene::default();
    scene.begin_view_chunk(view_id);
    scene.insert_primitive(test_quad(test_bounds(30.0, 0.0, 10.0, 10.0)));
    scene.end_view_chunk();
    scene.finish_with_previous_scene(&previous_scene);

    assert_eq!(scene.damage_rects.len(), 2);
    assert!(
        scene
            .damage_rects
            .contains(&test_bounds(0.0, 0.0, 10.0, 10.0))
    );
    assert!(
        scene
            .damage_rects
            .contains(&test_bounds(30.0, 0.0, 10.0, 10.0))
    );
}

#[test]
fn retained_scene_segment_ids_survive_frame_reconciliation() {
    let view_id = EntityId::from(1);
    let mut previous_scene = Scene::default();

    previous_scene.begin_view_chunk(view_id);
    previous_scene.insert_primitive(test_quad(test_bounds(0.0, 0.0, 10.0, 10.0)));
    previous_scene.end_view_chunk();
    previous_scene.finish();
    let previous_segment = previous_scene.segment_id_for_view(view_id).unwrap();

    let mut scene = Scene::default();
    scene.begin_view_chunk(view_id);
    scene.mark_chunk_clean(view_id);
    scene.replay(0..previous_scene.len(), &previous_scene);
    scene.end_view_chunk();
    scene.finish_with_previous_scene(&previous_scene);

    assert_eq!(scene.segment_id_for_view(view_id), Some(previous_segment));
    assert_eq!(scene.dirty_chunk_count(), 0);
}

#[test]
fn retained_scene_segments_follow_views_when_reordered() {
    let first_view = EntityId::from(1);
    let second_view = EntityId::from(2);
    let mut previous_scene = Scene::default();

    previous_scene.begin_view_chunk(first_view);
    previous_scene.insert_primitive(test_quad(test_bounds(0.0, 0.0, 10.0, 10.0)));
    previous_scene.end_view_chunk();
    previous_scene.begin_view_chunk(second_view);
    previous_scene.insert_primitive(test_quad(test_bounds(10.0, 0.0, 10.0, 10.0)));
    previous_scene.end_view_chunk();
    previous_scene.finish();
    let first_segment = previous_scene.segment_id_for_view(first_view).unwrap();
    let second_segment = previous_scene.segment_id_for_view(second_view).unwrap();

    let mut scene = Scene::default();
    scene.begin_view_chunk(second_view);
    scene.mark_chunk_clean(second_view);
    scene.replay(1..previous_scene.len(), &previous_scene);
    scene.end_view_chunk();
    scene.begin_view_chunk(first_view);
    scene.mark_chunk_clean(first_view);
    scene.replay(0..1, &previous_scene);
    scene.end_view_chunk();
    scene.finish_with_previous_scene(&previous_scene);

    assert_eq!(scene.segment_id_for_view(first_view), Some(first_segment));
    assert_eq!(scene.segment_id_for_view(second_view), Some(second_segment));
}

#[test]
fn removed_retained_scene_segments_are_not_carried_forward() {
    let removed_view = EntityId::from(1);
    let retained_view = EntityId::from(2);
    let mut previous_scene = Scene::default();

    previous_scene.begin_view_chunk(removed_view);
    previous_scene.insert_primitive(test_quad(test_bounds(0.0, 0.0, 10.0, 10.0)));
    previous_scene.end_view_chunk();
    previous_scene.begin_view_chunk(retained_view);
    previous_scene.insert_primitive(test_quad(test_bounds(10.0, 0.0, 10.0, 10.0)));
    previous_scene.end_view_chunk();
    previous_scene.finish();
    let retained_segment = previous_scene.segment_id_for_view(retained_view).unwrap();

    let mut scene = Scene::default();
    scene.begin_view_chunk(retained_view);
    scene.mark_chunk_clean(retained_view);
    scene.replay(1..previous_scene.len(), &previous_scene);
    scene.end_view_chunk();
    scene.finish_with_previous_scene(&previous_scene);

    assert_eq!(scene.segment_id_for_view(removed_view), None);
    assert_eq!(
        scene.segment_id_for_view(retained_view),
        Some(retained_segment)
    );
}
