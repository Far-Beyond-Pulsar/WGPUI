//! Compatibility crate for the unchanged examples.
//!
//! The public `gpui` contract is supplied by the legacy implementation while
//! the 2.0 crates are developed alongside it. Re-exporting the complete
//! implementation is intentional: it preserves event dispatch, hit testing,
//! asset loading, native window ownership, and GPU adapters instead of
//! replacing those behaviors with compile-only compatibility shims.

pub use gpui_legacy::*;

/// The in-progress 2.0 implementation remains available to compatibility
/// tests and downstream migration work without changing the legacy contract.
pub mod wgpui2 {
    pub use wgpui_core as core;
    pub use wgpui_layout as layout;
    pub use wgpui_text as text;
    pub use wgpui_widgets as widgets;
    pub use wgpui_wgpu as wgpu;
}
