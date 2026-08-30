use crate::{BackgroundExecutor, Task};
use std::{
    future::Future,
    pin::Pin,
    sync::atomic::{AtomicUsize, Ordering::SeqCst},
    task,
    time::Duration,
};

#[cfg(not(target_family = "wasm"))]
pub use util::*;

#[cfg(target_family = "wasm")]
pub use wasm_compat::*;

/// Emit a panic in debug builds, otherwise log the error.
/// Available at the crate root as `debug_panic!`.
#[macro_export]
macro_rules! debug_panic {
    ( $($fmt_arg:tt)* ) => {
        if cfg!(debug_assertions) {
            panic!( $($fmt_arg)* );
        } else {
            log::error!( $($fmt_arg)* );
        }
    };
}

#[cfg(target_family = "wasm")]
#[allow(missing_docs, dead_code)]
mod wasm_compat {
    pub mod arc_cow {
        use std::sync::Arc;

        pub enum ArcCow<'a, T: 'a> {
            Borrowed(&'a T),
            Owned(Arc<T>),
        }

        impl<'a, T> Clone for ArcCow<'a, T> {
            fn clone(&self) -> Self {
                match self {
                    ArcCow::Borrowed(v) => ArcCow::Borrowed(v),
                    ArcCow::Owned(v) => ArcCow::Owned(v.clone()),
                }
            }
        }

        impl<'a, T> std::ops::Deref for ArcCow<'a, T> {
            type Target = T;
            fn deref(&self) -> &T {
                match self {
                    ArcCow::Borrowed(v) => v,
                    ArcCow::Owned(v) => v,
                }
            }
        }
    }

    pub fn post_inc(x: &mut usize) -> usize {
        let v = *x;
        *x += 1;
        v
    }

    pub trait ResultExt<T, E> {
        fn log_err(self) -> std::option::Option<T>;
    }

    impl<T, E: std::fmt::Debug> ResultExt<T, E> for std::result::Result<T, E> {
        fn log_err(self) -> std::option::Option<T> {
            match self {
                Ok(val) => Some(val),
                Err(error) => {
                    log::error!("{:?}", error);
                    None
                }
            }
        }
    }

    pub trait TryFutureExt: futures::Future + Sized {
        fn log_tracked_err(
            self,
            location: std::panic::Location<'static>,
        ) -> impl futures::Future<Output = Self::Output>;
    }

    impl<T, E: std::fmt::Debug, F: futures::Future<Output = std::result::Result<T, E>>> TryFutureExt
        for F
    {
        fn log_tracked_err(
            self,
            location: std::panic::Location<'static>,
        ) -> impl futures::Future<Output = Self::Output> {
            let file: &'static str = location.file();
            let line: u32 = location.line();
            futures::FutureExt::map(self, move |result| {
                result.map_err(|error| {
                    log::error!("{}:{}: {:?}", file, line, error);
                    error
                })
            })
        }
    }

    pub struct Deferred<F: FnOnce()>(Option<F>);

    impl<F: FnOnce()> Deferred<F> {
        pub fn new(f: F) -> Self {
            Self(Some(f))
        }
    }

    impl<F: FnOnce()> Drop for Deferred<F> {
        fn drop(&mut self) {
            if let Some(f) = self.0.take() {
                f();
            }
        }
    }

    pub fn measure<T>(label: &str, f: impl FnOnce() -> T) -> T {
        let start = crate::time_ext::Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        log::info!("{} took {:?}", label, elapsed);
        result
    }

    pub fn defer(f: impl FnOnce() + 'static) -> Deferred<impl FnOnce()> {
        Deferred::new(f)
    }
}

#[allow(missing_docs)]
pub trait FluentBuilder {
    fn map<U>(self, f: impl FnOnce(Self) -> U) -> U
    where
        Self: Sized,
    {
        f(self)
    }

    fn when(self, condition: bool, then: impl FnOnce(Self) -> Self) -> Self
    where
        Self: Sized,
    {
        self.map(|this| if condition { then(this) } else { this })
    }

    fn when_else(
        self,
        condition: bool,
        then: impl FnOnce(Self) -> Self,
        else_fn: impl FnOnce(Self) -> Self,
    ) -> Self
    where
        Self: Sized,
    {
        self.map(|this| if condition { then(this) } else { else_fn(this) })
    }

    fn when_some<T>(self, option: Option<T>, then: impl FnOnce(Self, T) -> Self) -> Self
    where
        Self: Sized,
    {
        self.map(|this| {
            if let Some(value) = option {
                then(this, value)
            } else {
                this
            }
        })
    }

    fn when_none<T>(self, option: &Option<T>, then: impl FnOnce(Self) -> Self) -> Self
    where
        Self: Sized,
    {
        self.map(|this| if option.is_some() { this } else { then(this) })
    }
}

#[allow(missing_docs)]
pub trait FutureExt {
    fn with_timeout(self, timeout: Duration, executor: &BackgroundExecutor) -> WithTimeout<Self>
    where
        Self: Sized;
}

impl<T: Future> FutureExt for T {
    fn with_timeout(self, timeout: Duration, executor: &BackgroundExecutor) -> WithTimeout<Self>
    where
        Self: Sized,
    {
        WithTimeout {
            future: self,
            timer: executor.timer(timeout),
        }
    }
}

#[pin_project::pin_project]
pub struct WithTimeout<T> {
    #[pin]
    future: T,
    #[pin]
    timer: Task<()>,
}

#[derive(Debug, thiserror::Error)]
#[error("Timed out before future resolved")]
#[allow(missing_docs)]
pub struct Timeout;

impl<T: Future> Future for WithTimeout<T> {
    type Output = Result<T::Output, Timeout>;

    fn poll(self: Pin<&mut Self>, cx: &mut task::Context) -> task::Poll<Self::Output> {
        let this = self.project();

        if let task::Poll::Ready(output) = this.future.poll(cx) {
            task::Poll::Ready(Ok(output))
        } else if this.timer.poll(cx).is_ready() {
            task::Poll::Ready(Err(Timeout))
        } else {
            task::Poll::Pending
        }
    }
}

/// Await `f`, giving up after `timeout`.
///
/// Returns `Ok` with the future's output if it completed in time, or `Err(())`
/// if the timer fired first.
#[cfg(all(any(test, feature = "test-support"), not(target_family = "wasm")))]
pub async fn smol_timeout<F, T>(timeout: Duration, f: F) -> Result<T, ()>
where
    F: Future<Output = T>,
{
    let timer = async {
        smol::Timer::after(timeout).await;
        Err(())
    };
    let future = async move { Ok(f.await) };
    smol::future::FutureExt::race(timer, future).await
}

pub(crate) fn atomic_incr_if_not_zero(counter: &AtomicUsize) -> usize {
    let mut loaded = counter.load(SeqCst);
    loop {
        if loaded == 0 {
            return 0;
        }
        match counter.compare_exchange_weak(loaded, loaded + 1, SeqCst, SeqCst) {
            Ok(x) => return x + 1,
            Err(actual) => loaded = actual,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::TestAppContext;

    use super::*;

    #[gpui::test]
    async fn test_with_timeout(cx: &mut TestAppContext) {
        Task::ready(())
            .with_timeout(Duration::from_secs(1), &cx.executor())
            .await
            .expect("Timeout should be noop");

        let long_duration = Duration::from_secs(6000);
        let short_duration = Duration::from_secs(1);
        cx.executor()
            .timer(long_duration)
            .with_timeout(short_duration, &cx.executor())
            .await
            .expect_err("timeout should have triggered");

        let fut = cx
            .executor()
            .timer(long_duration)
            .with_timeout(short_duration, &cx.executor());
        cx.executor().advance_clock(short_duration * 2);
        futures::FutureExt::now_or_never(fut)
            .unwrap_or_else(|| panic!("timeout should have triggered"))
            .expect_err("timeout");
    }
}
