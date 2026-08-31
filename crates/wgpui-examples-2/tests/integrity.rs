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
