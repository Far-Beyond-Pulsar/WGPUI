const COMPAT_MANIFEST: &str = include_str!("../Cargo.toml");
const COMPAT_FACADE: &str = include_str!("../src/gpui.rs");
const LOCKFILE: &str = include_str!("../../../Cargo.lock");

fn compat_package_block(lockfile: &str) -> &str {
    let marker = "name = \"wgpui-compat\"";
    let marker_start = lockfile
        .find(marker)
        .expect("Cargo.lock must contain wgpui-compat");
    let block_start = lockfile[..marker_start]
        .rfind("[[package]]")
        .expect("wgpui-compat must be a package entry");
    let block_end = lockfile[marker_start..]
        .find("\n[[package]]")
        .map(|offset| marker_start + offset)
        .unwrap_or(lockfile.len());
    &lockfile[block_start..block_end]
}

#[test]
fn compatibility_facade_has_no_legacy_dependency_or_reexport() {
    assert!(!COMPAT_MANIFEST.contains("gpui-ce"));
    assert!(!COMPAT_MANIFEST.contains("gpui_legacy"));
    assert!(!COMPAT_FACADE.contains("gpui-ce"));
    assert!(!COMPAT_FACADE.contains("gpui_legacy"));
    let package = compat_package_block(LOCKFILE);
    assert!(!package.contains("gpui-ce"));
    assert!(!package.contains("gpui_legacy"));
}
