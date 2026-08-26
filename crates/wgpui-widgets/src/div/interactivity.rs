//! `Interactivity` state-and-paint engine — today's single largest block in
//! `src/elements/div.rs` (~1,140 lines, interleaving style application,
//! hitbox/dispatch registration, scroll handling, and layer paint), split
//! into `style`/`hitbox`/`layer_paint` here.
//! See docs/gpu-native-architecture.md §3.4.
#![allow(dead_code)]

pub mod hitbox;
pub mod layer_paint;
pub mod style;
