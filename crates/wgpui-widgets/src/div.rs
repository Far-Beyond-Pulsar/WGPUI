//! `Div`, `DivFrameState`/`DivPrepaintState` — the small remainder once
//! `div.rs`'s four seams (event-binding, interactivity, the `Element` impl,
//! scroll/click retained state) move to their own files.
//! See docs/gpu-native-architecture.md §3.4.
#![allow(dead_code)]

pub mod diff;
pub mod events;
pub mod interactivity;
pub mod scroll_state;

/// Placeholder for `Div`'s own `Element` impl plus reconciliation
/// fingerprint. Empty at Phase 0.
pub struct Div;
