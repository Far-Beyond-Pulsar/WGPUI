use wgpui_devtools::{CaptureBundle, CaptureError, ReferenceViewer};

const FIXTURE: &str = include_str!("../fixtures/reference_capture.json");

#[test]
fn reference_fixture_covers_the_external_consumer_sections() {
    let capture = CaptureBundle::from_json(FIXTURE).expect("reference fixture is valid");
    assert_eq!(capture.schema_version, 1);
    assert!(capture.capture.frozen_after_present);
    assert!(capture.element_tree.is_available());
    assert!(capture.flamegraph.is_available());
    assert!(capture.timeline.is_available());
    assert!(capture.memory.is_available());
    assert!(capture.listeners.is_available());
    assert!(capture.damage.is_available());
    assert!(capture.tiles.is_available());
    assert!(capture.resources.is_available());
    assert!(!capture.network.is_available());

    let viewer = ReferenceViewer::new(capture);
    let rendered = viewer.render();
    for section in [
        "element_tree: available",
        "flamegraph: available",
        "timeline: available",
        "memory: available",
        "listeners: available",
        "damage: available",
        "tiles: available",
        "resources: available",
        "network: unavailable (network capture was not armed for this frame)",
    ] {
        assert!(
            rendered.contains(section),
            "missing viewer section: {section}"
        );
    }
}

#[test]
fn unknown_json_fields_are_ignored_for_forward_compatibility() {
    let mut value: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture JSON");
    value["future_field"] = serde_json::json!({"future": true});
    let capture = CaptureBundle::from_json(&serde_json::to_string(&value).expect("JSON"))
        .expect("unknown fields do not break an older consumer");
    assert_eq!(capture.capture.frame_id, 42);
}

#[test]
fn framed_json_is_independent_of_the_renderer_process() {
    let capture = CaptureBundle::from_json(FIXTURE).expect("reference fixture is valid");
    let frame = capture.to_framed_json().expect("encode framed JSON");
    let decoded = ReferenceViewer::from_framed_json(&frame).expect("decode framed JSON");
    assert_eq!(decoded.capture(), &capture);

    assert_eq!(
        CaptureBundle::from_framed_json(b"not a capture"),
        Err(CaptureError::InvalidFrame("missing header"))
    );
}

#[test]
fn viewer_loads_a_frozen_json_file_without_a_live_renderer() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("reference_capture.json");
    let viewer = ReferenceViewer::from_path(path).expect("load capture file");
    assert!(viewer
        .render()
        .contains("schema=1 capture=reference-capture frame=42"));
}
