//! Backend-neutral instrumentation hooks.

use std::sync::Arc;

/// A no-op instrumentation implementation.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopHooks;

/// Hooks used by core and backend frame assembly.
pub trait InstrumentationHooks: Send + Sync {
    /// Starts a named CPU span and returns an opaque token.
    fn begin_span(&self, name: &'static str) -> Option<u64>;
    /// Completes a CPU span.
    fn end_span(&self, token: u64);
    /// Adds to a named counter.
    fn counter(&self, name: &'static str, amount: u64);
    /// Notifies the implementation that a frame was presented.
    fn frame_presented(&self);
    /// Records a backend timestamp pair when supported.
    fn gpu_timestamp(&self, _name: &'static str, _start: u64, _end: u64) {}
}

impl InstrumentationHooks for NoopHooks {
    fn begin_span(&self, _name: &'static str) -> Option<u64> {
        None
    }
    fn end_span(&self, _token: u64) {}
    fn counter(&self, _name: &'static str, _amount: u64) {}
    fn frame_presented(&self) {}
}

/// A shared hook handle suitable for frame assembly.
pub type SharedHooks = Arc<dyn InstrumentationHooks>;

/// A CPU span that closes itself when dropped.
pub struct Span<'a> {
    hooks: &'a dyn InstrumentationHooks,
    token: Option<u64>,
}
impl<'a> Span<'a> {
    /// Begins a span, or creates an inert guard when instrumentation is off.
    pub fn new(hooks: &'a dyn InstrumentationHooks, name: &'static str) -> Self {
        Self {
            hooks,
            token: hooks.begin_span(name),
        }
    }
}
impl Drop for Span<'_> {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            self.hooks.end_span(token);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Hooks(std::sync::Mutex<Vec<&'static str>>);
    impl InstrumentationHooks for Hooks {
        fn begin_span(&self, name: &'static str) -> Option<u64> {
            self.0.lock().expect("test mutex poisoned").push(name);
            Some(1)
        }
        fn end_span(&self, _token: u64) {}
        fn counter(&self, _name: &'static str, _amount: u64) {}
        fn frame_presented(&self) {}
    }
    #[test]
    fn span_calls_begin() {
        let hooks = Hooks::default();
        let _span = Span::new(&hooks, "frame");
        assert_eq!(
            hooks.0.lock().expect("test mutex poisoned").as_slice(),
            &["frame"]
        );
    }
}
