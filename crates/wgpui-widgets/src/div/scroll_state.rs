//! Shared native scroll handle used by scrollable elements.
//!
//! The handle has one owner so `div().track_scroll(...)`, lists, and the public
//! `wgpui::ScrollHandle` cannot silently diverge in clamping or wheel behavior.

pub use crate::scroll::ScrollHandle;
