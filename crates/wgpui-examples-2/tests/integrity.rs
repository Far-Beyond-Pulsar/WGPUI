use std::path::Path;

#[test]
fn copied_tree_and_native_dependency_are_intact() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let examples = manifest.join("examples");
    assert!(examples.join("legacy/hello_world.rs").is_file());
    assert!(examples.join("learn/emoji_display.rs").is_file());
    assert!(examples.join("legacy/image/black-cat-typing.gif").is_file());
    assert!(examples.join("legacy/svg/dragon.svg").is_file());

    let cargo_manifest = std::fs::read_to_string(manifest.join("Cargo.toml"))
        .expect("examples manifest must be readable");
    assert!(cargo_manifest.contains("wgpui = { path = \"../wgpui\" }"));
    assert!(!cargo_manifest.contains("gpui-ce"));
    assert!(!cargo_manifest.contains("gpui_legacy"));
    assert!(!cargo_manifest.contains("wgpui-compat"));
}

#[test]
fn migrated_text_and_layout_examples_use_the_native_render_contract() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    for relative_path in [
        "examples/learn/text.rs",
        "examples/learn/layout.rs",
        "examples/text_gradients.rs",
        "examples/bench/shadow.rs",
        "examples/bench/pattern.rs",
    ] {
        let source = std::fs::read_to_string(manifest.join(relative_path))
            .expect("migrated example must be readable");
        assert!(
            source.contains("fn render(&mut self) -> impl IntoElement + 'static"),
            "{relative_path} must implement native Render"
        );
        assert!(
            !source.contains("fn render(&mut self, window")
                && !source.contains("fn render(&mut self, _window"),
            "{relative_path} must not retain the legacy context-bearing Render signature"
        );
    }
}
