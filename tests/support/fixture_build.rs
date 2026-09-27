#![allow(dead_code)]
//! Builds and locates the `tests/dll_fixture` plugin DLL.

use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

/// The fixture DLL, built with the same profile as this test binary: plugins
/// must match the host's profile, since `cfg(debug_assertions)` fields change
/// struct layouts that plugin views share with the host.
pub fn fixture_path() -> &'static Path {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let target_dir = manifest_dir.join("target").join("dll-fixture");
        let release = !cfg!(debug_assertions);
        let mut command = Command::new(option_env!("CARGO").unwrap_or("cargo"));
        command
            .arg("build")
            .arg("--manifest-path")
            .arg(manifest_dir.join("tests").join("dll_fixture").join("Cargo.toml"))
            .arg("--target-dir")
            .arg(&target_dir);
        if release {
            command.arg("--release");
        }
        let status = command
            .status()
            .expect("failed to run cargo to build the DLL fixture");
        assert!(status.success(), "building the DLL fixture failed");
        target_dir
            .join(if release { "release" } else { "debug" })
            .join(libloading::library_filename("gpui_dll_fixture"))
    })
}

/// A second file with the same contents, so the OS maps it as a separate
/// module with its own statics: a second, independent plugin copy of gpui.
pub fn fixture_copy_path() -> &'static Path {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let original = fixture_path();
        let copy = original.with_file_name(libloading::library_filename("gpui_dll_fixture_copy"));
        std::fs::copy(original, &copy).expect("failed to copy the DLL fixture");
        copy
    })
}

#[cfg(windows)]
pub fn is_loaded(path: &Path) -> bool {
    match libloading::os::windows::Library::open_already_loaded(path) {
        Ok(library) => {
            library.close().expect("failed to release a probe handle");
            true
        }
        Err(_) => false,
    }
}

#[cfg(unix)]
pub fn is_loaded(path: &Path) -> bool {
    use libloading::os::unix::{Library, RTLD_LAZY};
    const RTLD_NOLOAD: std::ffi::c_int = if cfg!(target_os = "macos") { 0x10 } else { 0x4 };
    match unsafe { Library::open(Some(path), RTLD_LAZY | RTLD_NOLOAD) } {
        Ok(library) => {
            library.close().expect("failed to release a probe handle");
            true
        }
        Err(_) => false,
    }
}

#[cfg(windows)]
pub fn process_handle_count() -> u32 {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn GetProcessHandleCount(process: isize, count: *mut u32) -> i32;
    }
    let mut count = 0;
    let succeeded = unsafe { GetProcessHandleCount(GetCurrentProcess(), &mut count) };
    assert!(succeeded != 0, "GetProcessHandleCount failed");
    count
}

/// Private (committed, non-shared) bytes of this process, as the OS sees
/// them: catches leaks from any allocator, not just the tagged ones.
#[cfg(windows)]
pub fn process_private_bytes() -> u64 {
    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCountersEx {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        #[link_name = "K32GetProcessMemoryInfo"]
        fn GetProcessMemoryInfo(process: isize, counters: *mut ProcessMemoryCountersEx, size: u32) -> i32;
    }
    let mut counters = ProcessMemoryCountersEx {
        cb: size_of::<ProcessMemoryCountersEx>() as u32,
        ..Default::default()
    };
    let succeeded = unsafe {
        GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb)
    };
    assert!(succeeded != 0, "GetProcessMemoryInfo failed");
    counters.private_usage as u64
}
