//! What the UI thread is running, for the embedder.
//!
//! The UI thread should only build and update views; anything else (disk,
//! network, a process, a lock another thread holds, heavy compute) belongs on
//! another thread that sends its result back. This module is how an embedder
//! holds the framework to that:
//!
//! * The framework brackets each piece of work it runs on the UI thread with
//!   [`enter`]: a winit event, a foreground future (named by the `file:line`
//!   that spawned it), a view's `render` (named by its type), an action, a
//!   click or key listener (named by the listener's type, which is the
//!   function that wrote the closure).
//! * [`current`] says, from any thread, what the UI thread is inside right
//!   now, outer to inner. A watchdog that sees the UI stop can name the
//!   callback that stopped it.
//! * When a piece of work runs longer than the budget given to [`install`],
//!   the hook is called with its path and duration. Only the innermost slow
//!   piece is reported: a parent is reported only for the time its reported
//!   children do not explain.
//!
//! Nothing happens until [`install`] is called, from the UI thread; until
//! then [`enter`] is one atomic load.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::panic::Location;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// One piece of work on the UI thread.
#[derive(Clone, Copy, Debug)]
pub enum UiWork {
    /// A fixed stage of the event loop (`Main: WindowEvent::MouseInput`).
    Stage(&'static str),
    /// A foreground future, by the `file:line` that spawned it.
    Spawned(&'static Location<'static>),
    /// A foreground future whose spawn site was not recorded.
    SpawnedUntracked,
    /// A view's `render`, by the view's type.
    Render(&'static str),
    /// An action handler, by the action's name.
    Action(&'static str),
    /// An input listener: what kind, and the listener's type.
    Listener(&'static str, &'static str),
}

impl std::fmt::Display for UiWork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stage(name) => f.write_str(name),
            Self::Spawned(location) => write!(f, "foreground future spawned at {location}"),
            Self::SpawnedUntracked => f.write_str("foreground future (spawn site not recorded)"),
            Self::Render(view) => write!(f, "render {}", short_type_name(view)),
            Self::Action(name) => write!(f, "action {name}"),
            Self::Listener(kind, listener) => {
                write!(f, "{kind} listener {}", short_type_name(listener))
            }
        }
    }
}

/// A type name cut to a readable length. The generic arguments stay: for a
/// listener made with `cx.listener` they name the view and the callback.
fn short_type_name(name: &str) -> &str {
    const MAX: usize = 240;
    match name.char_indices().nth(MAX) {
        Some((cut, _)) => &name[..cut],
        None => name,
    }
}

/// What the embedder installed.
struct Hook {
    budget: Duration,
    on_slow: fn(String, Duration),
    ui_thread: std::thread::ThreadId,
}

static HOOK: OnceLock<Hook> = OnceLock::new();

/// The UI thread's stack, readable from any thread.
static PUBLISHED: Mutex<Vec<UiWork>> = Mutex::new(Vec::new());

struct Frame {
    start: Instant,
    /// Time of children already reported as slow.
    reported: Duration,
}

thread_local! {
    static STACK: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) };
}

/// Start watching the calling thread, the UI thread. `on_slow` is called on
/// the UI thread with a piece of work's path (outer to inner, `" > "`
/// separated) and duration when it ran for `budget` or longer. Only the
/// first call counts.
pub fn install(budget: Duration, on_slow: fn(String, Duration)) {
    let _ = HOOK.set(Hook {
        budget,
        on_slow,
        ui_thread: std::thread::current().id(),
    });
}

/// What the UI thread is running right now, outer to inner, or `None` when
/// it is between pieces of work (waiting for the OS) or nothing is
/// installed.
pub fn current() -> Option<String> {
    let stack = PUBLISHED.lock();
    if stack.is_empty() {
        return None;
    }
    Some(path(&stack))
}

fn path(stack: &[UiWork]) -> String {
    stack
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" > ")
}

/// Marks the end of a piece of work when dropped.
#[must_use = "the work ends when the guard is dropped"]
pub struct UiWorkGuard {
    active: bool,
    _not_send: PhantomData<*const ()>,
}

/// Bracket `work` until the returned guard drops. Off the UI thread, or
/// before [`install`], it does nothing.
#[inline]
pub fn enter(work: UiWork) -> UiWorkGuard {
    let active = HOOK
        .get()
        .is_some_and(|hook| hook.ui_thread == std::thread::current().id());
    if active {
        PUBLISHED.lock().push(work);
        STACK.with(|stack| {
            stack.borrow_mut().push(Frame {
                start: Instant::now(),
                reported: Duration::ZERO,
            })
        });
    }
    UiWorkGuard {
        active,
        _not_send: PhantomData,
    }
}

impl Drop for UiWorkGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let Some(hook) = HOOK.get() else { return };
        let Some(frame) = STACK.with(|stack| stack.borrow_mut().pop()) else {
            return;
        };
        let elapsed = frame.start.elapsed();
        let unexplained = elapsed.saturating_sub(frame.reported);
        let slow = unexplained >= hook.budget;
        let report = slow.then(|| path(&PUBLISHED.lock()));
        PUBLISHED.lock().pop();
        if slow {
            // The parent's own time excludes what is reported here.
            STACK.with(|stack| {
                if let Some(parent) = stack.borrow_mut().last_mut() {
                    parent.reported += elapsed;
                }
            });
        } else {
            STACK.with(|stack| {
                if let Some(parent) = stack.borrow_mut().last_mut() {
                    parent.reported += frame.reported;
                }
            });
        }
        if let Some(report) = report {
            (hook.on_slow)(report, elapsed);
        }
    }
}

/// The name a listener is reported by: its own type, which for a closure is
/// the function that wrote it.
pub(crate) fn listener_label<L: ?Sized>() -> &'static str {
    std::any::type_name::<L>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closure_types_are_shortened_to_their_function() {
        assert_eq!(
            short_type_name("app::panel::Panel::render::{{closure}}"),
            "app::panel::Panel::render::{{closure}}"
        );
        assert_eq!(short_type_name(&"x".repeat(500)).len(), 240);
    }
}
