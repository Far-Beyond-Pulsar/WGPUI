use std::time::{Duration, Instant};
use std::future::Future;

/// A lightweight awaitable delay for foreground tasks.
pub struct Timer;

#[derive(Copy, Clone, Debug, Default)]
pub struct BackgroundExecutor;

impl BackgroundExecutor {
    pub fn timer(&self, duration: Duration) -> impl Future<Output = ()> {
        Timer::after(duration)
    }
}

impl Timer {
    /// Complete after `duration` without blocking the foreground executor.
    pub async fn after(duration: Duration) {
        let (sender, receiver) = futures::channel::oneshot::channel();
        let spawned = std::thread::Builder::new()
            .name("wgpui-timer".to_string())
            .spawn(move || {
                std::thread::sleep(duration);
                if sender.send(()).is_err() {
                    return;
                }
            });
        if spawned.is_err() {
            return;
        }
        if receiver.await.is_err() {
            return;
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TimerId(u64);
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TimerHandle {
    id: TimerId,
}
impl TimerHandle {
    pub const fn id(self) -> TimerId {
        self.id
    }
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TimerState {
    Pending,
    Fired,
    Cancelled,
}
#[derive(Debug)]
struct Entry {
    id: TimerId,
    deadline: Instant,
    state: TimerState,
}
#[derive(Debug, Default)]
pub struct TimerScheduler {
    entries: Vec<Entry>,
    next_id: u64,
}
impl TimerScheduler {
    pub fn schedule(&mut self, now: Instant, delay: Duration) -> TimerHandle {
        self.next_id = self.next_id.wrapping_add(1);
        let id = TimerId(self.next_id);
        self.entries.push(Entry {
            id,
            deadline: now + delay,
            state: TimerState::Pending,
        });
        TimerHandle { id }
    }
    pub fn cancel(&mut self, timer: TimerHandle) -> bool {
        self.entries
            .iter_mut()
            .find(|entry| entry.id == timer.id && entry.state == TimerState::Pending)
            .map(|entry| {
                entry.state = TimerState::Cancelled;
                true
            })
            .unwrap_or(false)
    }
    pub fn take_due(&mut self, now: Instant) -> Vec<TimerId> {
        let mut due = Vec::new();
        for entry in &mut self.entries {
            if entry.state == TimerState::Pending && entry.deadline <= now {
                entry.state = TimerState::Fired;
                due.push(entry.id);
            }
        }
        due
    }
    pub fn next_deadline(&self) -> Option<Instant> {
        self.entries
            .iter()
            .filter(|entry| entry.state == TimerState::Pending)
            .map(|entry| entry.deadline)
            .min()
    }
    pub fn state(&self, timer: TimerHandle) -> Option<TimerState> {
        self.entries
            .iter()
            .find(|entry| entry.id == timer.id)
            .map(|entry| entry.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    #[test]
    fn cancellation_prevents_a_due_timer_from_firing() {
        let now = Instant::now();
        let mut timers = TimerScheduler::default();
        let cancelled = timers.schedule(now, Duration::from_secs(1));
        let fired = timers.schedule(now, Duration::from_secs(1));
        assert!(timers.cancel(cancelled));
        assert_eq!(
            timers.take_due(now + Duration::from_secs(1)),
            vec![fired.id]
        );
        assert_eq!(timers.state(cancelled), Some(TimerState::Cancelled));
    }

    #[test]
    fn awaitable_timer_completes_without_blocking_the_executor() {
        futures::executor::block_on(Timer::after(Duration::from_millis(1)));
    }
}
