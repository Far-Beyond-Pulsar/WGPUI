//! `window.rs`'s actual successors, one concern each — the targeted split
//! of today's 14,278-line, 450-method `impl Window` block.
//! See docs/gpu-native-architecture.md §1, §3.1.
#![allow(dead_code)]

pub mod animation;
pub mod dispatch;
pub mod focus;
pub mod hitbox;
pub mod input;
pub mod prompts;

/// Placeholder for `Window` struct assembly only — the split moves behavior
/// into `focus`/`hitbox`/`dispatch`/`input`/`animation`/`prompts`, this file
/// just wires them together.
pub struct Window;
