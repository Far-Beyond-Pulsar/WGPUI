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
use crate::reconcile::{ElementStateStore, StateKey, StateScope};

pub struct Window {
    state: ElementStateStore,
    frame: u64,
}
impl Default for Window {
    fn default() -> Self {
        Self::new()
    }
}
impl Window {
    pub fn new() -> Self {
        Self {
            state: ElementStateStore::new(),
            frame: 0,
        }
    }
    pub fn next_frame(&mut self) {
        self.frame = self.frame.wrapping_add(1);
    }
    pub fn use_state<T: 'static, R>(
        &mut self,
        scope: StateScope,
        initialise: impl FnOnce() -> T,
        access: impl FnOnce(&mut T) -> R,
    ) -> Option<R> {
        self.state
            .with_state(StateKey::new::<T>(scope), self.frame, initialise, access)
    }
    pub fn state_len(&self) -> usize {
        self.state.len()
    }
}
