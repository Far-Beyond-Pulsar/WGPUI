const ROOT_MANIFEST: &str = include_str!("../../../Cargo.toml");
const COMPAT_MANIFEST: &str = include_str!("../Cargo.toml");

fn example_names(manifest: &str) -> Vec<&str> {
    let mut in_example = false;
    let mut names = Vec::new();
    for line in manifest.lines() {
        if line.trim() == "[[example]]" {
            in_example = true;
        } else if in_example && line.trim_start().starts_with("name = ") {
            if let Some(name) = line.split('"').nth(1) {
                names.push(name);
            }
            in_example = false;
        }
    }
    names.sort_unstable();
    names
}

#[test]
fn every_declared_legacy_example_is_in_the_compatibility_probe() {
    assert_eq!(example_names(ROOT_MANIFEST), example_names(COMPAT_MANIFEST));
}
