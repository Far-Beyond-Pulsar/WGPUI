#![allow(clippy::disallowed_methods, reason = "build scripts are exempt")]

fn main() {
    // Without an explicit rerun-if-changed, cargo reruns this script (and so
    // recompiles the crate) whenever any file in the package changes,
    // including tests and docs.
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(gles)");
}
