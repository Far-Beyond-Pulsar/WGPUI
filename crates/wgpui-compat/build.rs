use std::fs;
use std::path::PathBuf;

const FORBIDDEN_NAMES: [&str; 2] = ["gpui-ce", "gpui_legacy"];

fn package_block<'a>(lockfile: &'a str, package_name: &str) -> Option<&'a str> {
    let marker = format!("name = \"{package_name}\"");
    let package_start = lockfile.find("[[package]]")?;
    let marker_start = lockfile[package_start..].find(&marker)? + package_start;
    let block_start = lockfile[..marker_start].rfind("[[package]]")?;
    let block_end = lockfile[marker_start..]
        .find("\n[[package]]")
        .map(|offset| marker_start + offset)
        .unwrap_or(lockfile.len());
    Some(&lockfile[block_start..block_end])
}

fn contains_forbidden_name(text: &str) -> bool {
    FORBIDDEN_NAMES.iter().any(|name| text.contains(name))
}

fn main() {
    let manifest_directory = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR must be set for the compatibility guard"),
    );
    let manifest_path = manifest_directory.join("Cargo.toml");
    let facade_path = manifest_directory.join("src/gpui.rs");
    let lockfile_path = manifest_directory.join("../../Cargo.lock");
    let manifest = fs::read_to_string(&manifest_path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", manifest_path.display()));
    let facade = fs::read_to_string(&facade_path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", facade_path.display()));
    let lockfile = fs::read_to_string(&lockfile_path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", lockfile_path.display()));

    if contains_forbidden_name(&manifest) || contains_forbidden_name(&facade) {
        panic!("wgpui-compat must not mention the legacy package or alias");
    }
    let compat_package = package_block(&lockfile, "wgpui-compat")
        .expect("Cargo.lock must contain the wgpui-compat package");
    if contains_forbidden_name(compat_package) {
        panic!("wgpui-compat has a legacy dependency in Cargo.lock");
    }

    println!("cargo:rerun-if-changed={}", manifest_path.display());
    println!("cargo:rerun-if-changed={}", facade_path.display());
    println!("cargo:rerun-if-changed={}", lockfile_path.display());
}
