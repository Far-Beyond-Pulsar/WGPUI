//! Replay adapter boundary for backend-owned captures.
pub use wgpui_core::diagnostics::{
    FrozenDamage, FrozenFrameReport, ReplayInputRecord, SingleFrameInput,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayViewport {
    pub width: u32,
    pub height: u32,
}
