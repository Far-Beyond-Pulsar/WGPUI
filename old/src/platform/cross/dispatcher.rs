use crate::{PlatformDispatcher, Priority, PriorityQueueSender, RealtimePriority, RunnableVariant};
use std::thread::ThreadId;
use winit::event_loop::EventLoopProxy;

#[cfg(not(target_family = "wasm"))]
use priority_threadpool::ThreadPool;

pub enum CrossEvent {
    WakeUp,
    SurfacePresent(winit::window::WindowId),
    SingleInstanceActivated,
    CloseWindow(winit::window::WindowId),
}

pub struct Dispatcher {
    main_thread_id: ThreadId,
    main_tx: PriorityQueueSender<RunnableVariant>,
    proxy: EventLoopProxy<CrossEvent>,
    #[cfg(not(target_family = "wasm"))]
    threadpool: ThreadPool<Priority>,
}

impl Dispatcher {
    pub fn new(
        main_tx: PriorityQueueSender<RunnableVariant>,
        proxy: EventLoopProxy<CrossEvent>,
    ) -> Self {
        Self {
            main_thread_id: std::thread::current().id(),
            main_tx,
            proxy,
            #[cfg(not(target_family = "wasm"))]
            threadpool: ThreadPool::new(num_cpus::get() * 8),
        }
    }
}

impl PlatformDispatcher for Dispatcher {
    fn is_main_thread(&self) -> bool {
        std::thread::current().id() == self.main_thread_id
    }

    fn dispatch(
        &self,
        runnable: RunnableVariant,
        _label: Option<crate::TaskLabel>,
        priority: Priority,
    ) {
        #[cfg(not(target_family = "wasm"))]
        match runnable {
            RunnableVariant::Meta(runnable) => self.threadpool.queue(&priority, runnable),
            RunnableVariant::Compat(runnable) => self.threadpool.queue(&priority, runnable),
        }
        #[cfg(target_family = "wasm")]
        {
            // On WASM, schedule on the main thread event loop instead of
            // running synchronously (which would block rendering).
            let _ = self.main_tx.send(priority, runnable);
            let _ = self.proxy.send_event(CrossEvent::WakeUp);
        }
    }

    fn dispatch_on_main_thread(&self, runnable: RunnableVariant, priority: Priority) {
        match self.main_tx.send(priority, runnable) {
            Ok(_) => {
                let _ = self.proxy.send_event(CrossEvent::WakeUp);
            }
            Err(runnable) => {
                std::mem::forget(runnable);
            }
        }
    }

    fn dispatch_after(&self, duration: std::time::Duration, runnable: RunnableVariant) {
        #[cfg(not(target_family = "wasm"))]
        match runnable {
            RunnableVariant::Meta(runnable) => {
                self.threadpool
                    .queue_delayed(&Priority::Low, duration, runnable);
            }
            RunnableVariant::Compat(runnable) => {
                self.threadpool
                    .queue_delayed(&Priority::Low, duration, runnable);
            }
        }
        #[cfg(target_family = "wasm")]
        {
            // On WASM, use setTimeout via a JS promise to schedule after delay
            let main_tx = self.main_tx.clone();
            let proxy = self.proxy.clone();
            let ms = duration.as_millis().min(u32::MAX as u128) as f64;
            wasm_bindgen_futures::spawn_local(async move {
                let promise = js_sys::Promise::new(&mut |resolve, _reject| {
                    let window = web_sys::window().unwrap();
                    let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
                        &resolve,
                        ms as i32,
                    );
                });
                let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
                let _ = main_tx.send(Priority::Low, runnable);
                let _ = proxy.send_event(CrossEvent::WakeUp);
            });
        }
    }

    fn spawn_realtime(&self, _priority: RealtimePriority, f: Box<dyn FnOnce() + Send>) {
        #[cfg(not(target_family = "wasm"))]
        std::thread::spawn(move || {
            f();
        });
        #[cfg(target_family = "wasm")]
        {
            // On WASM, schedule via microtask to avoid blocking the event loop
            wasm_bindgen_futures::spawn_local(async move {
                f();
            });
        }
    }
}

#[cfg(not(target_family = "wasm"))]
impl priority_threadpool::Priority for Priority {
    const COUNT: usize = 3;

    fn index(&self) -> usize {
        match self {
            Priority::High => 0,
            Priority::Medium => 1,
            Priority::Low => 2,
            _ => unreachable!(),
        }
    }
}
