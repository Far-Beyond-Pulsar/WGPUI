//! Exports the C APIs of the native libraries inside this library.
//!
//! With the editor's features unified, WGPUI's dependencies pull zlib (`libz-sys`, through
//! `flate2`) and tree-sitter into this library. A Rust dylib exports Rust symbols, not the C
//! symbols of the native libraries inside it, so a crate outside it that calls them
//! (`libgit2-sys` calls zlib, `mockaco-language` calls tree-sitter, in the editor and in every
//! plugin library) fails to link (`unresolved external symbol deflate`). On MSVC a linker
//! directive (`/EXPORT:name`) in the library's own object code exports them; this writes that
//! directive as a `.drectve` section. On macOS the same list is passed to the linker as exported symbols.
//!
//! `native_exports/<library>.txt` lists the functions (a line `name`, or `name,DATA`),
//! generated from each crate's Rust bindings (`libz-sys` 1.1.33 `src/lib.rs`, `tree-sitter`
//! 0.25.10 `binding_rust/bindings.rs`); a name the compiled library does not define (a feature
//! that is off) must be removed, the linker names it. Regenerate when a version changes.

use std::{env, fs, path::Path};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=native_exports");
    let out = Path::new(&env::var("OUT_DIR").unwrap()).join("native_exports.rs");
    // macOS: the system frameworks the crates inside this library call. Linking the library from their rlibs does not
    // carry a framework named by a dependency's own link attributes through (`io_surface`'s `_IOSurface*` were
    // undefined when the library was linked), so they are named here, on the library itself.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        for framework in ["IOSurface", "CoreVideo", "Metal", "QuartzCore"] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
    }
    // macOS: the same list, as exported symbols of the library (`-exported_symbol _name`, which adds to the
    // symbols rustc exports). Without it the library exports its Rust symbols only, and a plugin library that
    // links a crate calling these C functions (engine_fs calls SQLite through rusqlite) fails to link
    // (`Undefined symbols for architecture arm64: _sqlite3_bind_blob`).
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        let mut files: Vec<_> = fs::read_dir("native_exports")
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "txt"))
            .collect();
        files.sort();
        for file in files {
            for line in fs::read_to_string(file).unwrap().lines() {
                let name = line.trim().split(',').next().unwrap_or_default();
                if !name.is_empty() {
                    println!("cargo:rustc-link-arg=-Wl,-exported_symbol,_{name}");
                }
            }
        }
    }
    let windows_msvc = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    if !windows_msvc {
        fs::write(out, "").unwrap();
        return;
    }
    let mut files: Vec<_> = fs::read_dir("native_exports")
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "txt"))
        .collect();
    files.sort();
    let mut directive = String::new();
    for file in files {
        for line in fs::read_to_string(file).unwrap().lines() {
            let line = line.trim();
            if !line.is_empty() {
                directive.push_str(&format!(" /EXPORT:{line}"));
            }
        }
    }
    let len = directive.len();
    let source = format!(
        "#[used]\n#[link_section = \".drectve\"]\nstatic NATIVE_EXPORTS: [u8; {len}] = *b{directive:?};\n"
    );
    fs::write(out, source).unwrap();
}
