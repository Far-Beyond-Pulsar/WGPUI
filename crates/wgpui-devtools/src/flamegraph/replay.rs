//! Replay adapter boundary for backend-owned captures.
//!
//! The core replay report remains an internal implementation until its input
//! event wire format is made serializable. Keeping this adapter independent
//! lets the devtools crate expose the viewport contract without pretending
//! that live input events can already be exported safely.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayViewport {
    pub width: u32,
    pub height: u32,
}
