//! Coalescing a burst of resize events into at most one reconfiguration.
//! See docs/gpu-native-architecture.md §3.5.
//!
//! # Not the legacy detector, and the difference is the point
//!
//! `src/platform/cross/resize_detector.rs` answers a different question —
//! *"is the user still dragging the resize handle?"* — by polling global mouse
//! button state through `device_query`, so the legacy backend can hold off
//! expensive work until a drag ends. That is a policy decision about when to
//! re-lay-out, it needs a whole extra dependency to make it, and Phase 6 has no
//! layout to defer.
//!
//! What Phase 6 does need is the mechanical half underneath it: a window being
//! dragged emits `WindowEvent::Resized` far faster than frames are drawn, and
//! `wgpu::Surface::configure` is not a cheap call — it waits for the device to
//! go idle (the legacy `reconfigure_surface`'s own doc comment says so, and
//! guards it with an exclusive lock for exactly that reason). Calling it once
//! per event rather than once per frame is how a resize turns into a stall.
//!
//! So this type holds *the latest size seen* and nothing else, and the loop
//! takes it once, immediately before it acquires. Every size in between is
//! dropped on purpose: it was never going to be presented.
//!
//! The counters exist because "coalesced" is otherwise an intention rather than
//! a fact — [`ResizeDetector::events_seen`] minus
//! [`ResizeDetector::reconfigurations`] is the number of `configure` calls this
//! type actually removed, and Phase 6's resize evidence quotes it.

/// The latest size a window has been resized to, and what was dropped to get
/// there.
#[derive(Clone, Debug, Default)]
pub struct ResizeDetector {
    pending: Option<(u32, u32)>,
    applied: Option<(u32, u32)>,
    events_seen: u64,
    reconfigurations: u64,
    zero_sized: u64,
}

impl ResizeDetector {
    /// A detector for a window that has never been resized.
    pub fn new() -> ResizeDetector {
        ResizeDetector::default()
    }

    /// The size this detector believes the surface is currently configured at.
    ///
    /// Set by [`Self::seed`] at creation and by every [`Self::take_pending`]
    /// that hands a size out, so a resize event naming the size already in use
    /// — which Windows does emit, on restore and on a drag that ends where it
    /// started — is recognised as a no-op rather than paid for.
    pub fn applied(&self) -> Option<(u32, u32)> {
        self.applied
    }

    /// Record the size the surface was first configured at.
    pub fn seed(&mut self, width: u32, height: u32) {
        self.applied = Some((width, height));
    }

    /// Record one `WindowEvent::Resized`, replacing any earlier pending size.
    ///
    /// A zero width or height is counted and then discarded rather than stored:
    /// it is what Windows reports for a minimized window, and
    /// `Surface::configure` rejects a zero extent. Treating it as "no pending
    /// resize" is correct — there is nothing to present to — and the restore
    /// that follows carries its own event with the real size.
    pub fn on_resize_event(&mut self, width: u32, height: u32) {
        self.events_seen += 1;
        if width == 0 || height == 0 {
            self.zero_sized += 1;
            return;
        }
        self.pending = Some((width, height));
    }

    /// The size to reconfigure to, if it differs from the one in use.
    ///
    /// Consumes the pending size whether or not it is handed out, because a
    /// pending size equal to the applied one is genuinely resolved: nothing
    /// about the surface has to change for it to be correct.
    pub fn take_pending(&mut self) -> Option<(u32, u32)> {
        let pending = self.pending.take()?;
        if self.applied == Some(pending) {
            return None;
        }
        self.applied = Some(pending);
        self.reconfigurations += 1;
        Some(pending)
    }

    /// How many resize events were observed.
    pub fn events_seen(&self) -> u64 {
        self.events_seen
    }

    /// How many of those turned into an actual `Surface::configure`.
    pub fn reconfigurations(&self) -> u64 {
        self.reconfigurations
    }

    /// How many of those named a zero-sized window (minimized).
    pub fn zero_sized(&self) -> u64 {
        self.zero_sized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_burst_of_events_costs_one_reconfiguration() {
        let mut detector = ResizeDetector::new();
        detector.seed(640, 360);
        for width in 641..=700 {
            detector.on_resize_event(width, 360);
        }
        assert_eq!(detector.events_seen(), 60);
        assert_eq!(detector.take_pending(), Some((700, 360)));
        assert_eq!(detector.reconfigurations(), 1);
        assert_eq!(detector.take_pending(), None);
    }

    #[test]
    fn a_resize_back_to_the_configured_size_is_not_a_reconfiguration() {
        let mut detector = ResizeDetector::new();
        detector.seed(640, 360);
        detector.on_resize_event(640, 360);
        assert_eq!(detector.take_pending(), None);
        assert_eq!(detector.events_seen(), 1);
        assert_eq!(detector.reconfigurations(), 0);
    }

    #[test]
    fn a_minimize_is_counted_and_never_configured() {
        // Windows reports a minimized window as 0x0, and `Surface::configure`
        // rejects a zero extent — so this has to be dropped, not deferred.
        let mut detector = ResizeDetector::new();
        detector.seed(640, 360);
        detector.on_resize_event(0, 0);
        assert_eq!(detector.take_pending(), None);
        assert_eq!(detector.zero_sized(), 1);
        // ...and the restore that follows is an ordinary resize.
        detector.on_resize_event(640, 360);
        assert_eq!(detector.take_pending(), None);
        detector.on_resize_event(800, 600);
        assert_eq!(detector.take_pending(), Some((800, 600)));
    }

    #[test]
    fn shrinking_then_growing_reconfigures_both_ways() {
        // Down-then-up is the documented sharp edge in surface reconfiguration:
        // the shrink is the direction that frees swapchain images, and a
        // detector that only ever grew a high-water mark would silently skip
        // the second half.
        let mut detector = ResizeDetector::new();
        detector.seed(800, 600);
        detector.on_resize_event(200, 150);
        assert_eq!(detector.take_pending(), Some((200, 150)));
        detector.on_resize_event(1200, 900);
        assert_eq!(detector.take_pending(), Some((1200, 900)));
        assert_eq!(detector.reconfigurations(), 2);
    }
}
