use crate::{
    AnyWindowHandle, BackgroundExecutor, Bounds, ClipboardItem, CursorStyle, DummyKeyboardMapper,
    ForegroundExecutor, GpuSpecs, Keymap, Pixels, Platform, PlatformDisplay, PlatformInputHandler,
    PlatformKeyboardLayout, PlatformKeyboardMapper, PlatformTextSystem, PlatformWindow, Point,
    Priority, PriorityQueueReceiver, PriorityQueueSender, PromptButton, RequestFrameOptions,
    ResizeEdge, Result, RunnableVariant, Size, Task, WgpuOutputHandle, WgpuSurfaceHandle,
    WindowAppearance, WindowBackgroundAppearance, WindowBounds, WindowControlArea, WindowParams,
    platform::cross::{
        atlas::WgpuAtlas, render_context::WgpuContext, renderer::WgpuRenderer,
        text_system::CosmicTextSystem,
    },
    px,
};
use anyhow::anyhow;
use futures::channel::oneshot;
use parking_lot::Mutex;
use priority_threadpool::ThreadPool;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct HeadlessWindowId(u64);

enum HeadlessEvent {
    Runnable(RunnableVariant),
    RequestFrame(HeadlessWindowId, RequestFrameOptions),
    Resized(HeadlessWindowId, Size<Pixels>, f32),
    CloseWindow(HeadlessWindowId),
    Quit,
}

struct HeadlessDispatcher {
    main_thread_id: std::thread::ThreadId,
    main_tx: Arc<PriorityQueueSender<HeadlessEvent>>,
    threadpool: ThreadPool<Priority>,
}

impl HeadlessDispatcher {
    fn new(main_tx: Arc<PriorityQueueSender<HeadlessEvent>>) -> Self {
        Self {
            main_thread_id: std::thread::current().id(),
            main_tx,
            threadpool: ThreadPool::new(num_cpus::get().max(1) * 8),
        }
    }
}

impl crate::PlatformDispatcher for HeadlessDispatcher {
    fn get_all_timings(&self) -> Vec<crate::ThreadTaskTimings> {
        let global_thread_timings = crate::GLOBAL_THREAD_TIMINGS.lock();
        crate::ThreadTaskTimings::convert(&global_thread_timings)
    }

    fn get_current_thread_timings(&self) -> Vec<crate::TaskTiming> {
        crate::THREAD_TIMINGS.with(|timings| {
            let timings = timings.lock();
            let timings = &timings.timings;

            let mut collected = Vec::with_capacity(timings.len());
            let (first, second) = timings.as_slices();
            collected.extend_from_slice(first);
            collected.extend_from_slice(second);
            collected
        })
    }

    fn is_main_thread(&self) -> bool {
        std::thread::current().id() == self.main_thread_id
    }

    fn dispatch(
        &self,
        runnable: RunnableVariant,
        _label: Option<crate::TaskLabel>,
        priority: Priority,
    ) {
        match runnable {
            RunnableVariant::Meta(runnable) => self.threadpool.queue(&priority, runnable),
            RunnableVariant::Compat(runnable) => self.threadpool.queue(&priority, runnable),
        }
    }

    fn dispatch_on_main_thread(&self, runnable: RunnableVariant, priority: Priority) {
        let _ = self
            .main_tx
            .send(priority, HeadlessEvent::Runnable(runnable));
    }

    fn dispatch_after(&self, duration: Duration, runnable: RunnableVariant) {
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
    }

    fn spawn_realtime(&self, _priority: crate::RealtimePriority, f: Box<dyn FnOnce() + Send>) {
        std::thread::spawn(move || {
            f();
        });
    }
}

#[derive(Debug)]
struct HeadlessDisplay {
    id: crate::DisplayId,
    uuid: uuid::Uuid,
    bounds: Bounds<Pixels>,
}

impl HeadlessDisplay {
    fn new() -> Self {
        Self {
            id: crate::DisplayId(1),
            uuid: uuid::Uuid::new_v4(),
            bounds: Bounds::from_corners(Point::default(), Point::new(px(1920.0), px(1080.0))),
        }
    }
}

impl PlatformDisplay for HeadlessDisplay {
    fn id(&self) -> crate::DisplayId {
        self.id
    }

    fn uuid(&self) -> Result<uuid::Uuid> {
        Ok(self.uuid)
    }

    fn bounds(&self) -> Bounds<Pixels> {
        self.bounds
    }
}

#[derive(Default)]
struct HeadlessCallbacks {
    on_request_frame: RefCell<Option<Box<dyn FnMut(RequestFrameOptions)>>>,
    on_input: RefCell<Option<Box<dyn FnMut(crate::PlatformInput) -> crate::DispatchEventResult>>>,
    on_active_status_change: RefCell<Option<Box<dyn FnMut(bool)>>>,
    on_hover_status_change: RefCell<Option<Box<dyn FnMut(bool)>>>,
    on_resize: RefCell<Option<Box<dyn FnMut(Size<Pixels>, f32)>>>,
    on_moved: RefCell<Option<Box<dyn FnMut()>>>,
    on_should_close: RefCell<Option<Box<dyn FnMut() -> bool>>>,
    on_hit_test_window_control: RefCell<Option<Box<dyn FnMut() -> Option<WindowControlArea>>>>,
    on_close: RefCell<Option<Box<dyn FnOnce()>>>,
    on_appearance_changed: RefCell<Option<Box<dyn FnMut()>>>,
}

impl HeadlessCallbacks {
    fn invoke_mut<F: ?Sized>(slot: &RefCell<Option<Box<F>>>, callback: impl FnOnce(&mut F)) {
        let Some(mut slot_callback) = slot.borrow_mut().take() else {
            return;
        };
        callback(&mut slot_callback);
        *slot.borrow_mut() = Some(slot_callback);
    }

    fn invoke_once<F: ?Sized>(slot: &RefCell<Option<Box<F>>>, callback: impl FnOnce(Box<F>)) {
        let Some(slot_callback) = slot.borrow_mut().take() else {
            return;
        };
        callback(slot_callback);
    }
}

struct HeadlessWindowState {
    id: HeadlessWindowId,
    handle: AnyWindowHandle,
    bounds: RefCell<Bounds<Pixels>>,
    scale_factor: Cell<f32>,
    display: Rc<dyn PlatformDisplay>,
    sprite_atlas: Arc<WgpuAtlas>,
    wgpu_context: Arc<WgpuContext>,
    renderer: RefCell<WgpuRenderer>,
    output: WgpuOutputHandle,
    event_tx: Arc<PriorityQueueSender<HeadlessEvent>>,
    input_handler: RefCell<Option<PlatformInputHandler>>,
    callbacks: HeadlessCallbacks,
    title: RefCell<Option<String>>,
    app_id: RefCell<Option<String>>,
    edited: Cell<bool>,
    background_appearance: Cell<WindowBackgroundAppearance>,
    active: Cell<bool>,
}

impl HeadlessWindowState {
    fn new(
        id: HeadlessWindowId,
        handle: AnyWindowHandle,
        params: WindowParams,
        display: Rc<dyn PlatformDisplay>,
        wgpu_context: Arc<WgpuContext>,
        event_tx: Arc<PriorityQueueSender<HeadlessEvent>>,
    ) -> anyhow::Result<Self> {
        let sprite_atlas = Arc::new(WgpuAtlas::new(wgpu_context.clone()));
        let width = params.bounds.size.width.0.max(1.0).round() as u32;
        let height = params.bounds.size.height.0.max(1.0).round() as u32;
        let (renderer, output) = WgpuRenderer::new_offscreen(
            wgpu_context.clone(),
            sprite_atlas.clone(),
            width,
            height,
            4,
        )?;

        Ok(Self {
            id,
            handle,
            bounds: RefCell::new(params.bounds),
            scale_factor: Cell::new(1.0),
            display,
            sprite_atlas,
            wgpu_context,
            renderer: RefCell::new(renderer),
            output,
            event_tx,
            input_handler: RefCell::new(None),
            callbacks: HeadlessCallbacks::default(),
            title: RefCell::new(
                params
                    .titlebar
                    .and_then(|titlebar| titlebar.title.map(|title| title.to_string())),
            ),
            app_id: RefCell::new(None),
            edited: Cell::new(false),
            background_appearance: Cell::new(WindowBackgroundAppearance::Opaque),
            active: Cell::new(true),
        })
    }

    fn request_frame_internal(&self, options: RequestFrameOptions) {
        let _ = self.event_tx.send(
            Priority::High,
            HeadlessEvent::RequestFrame(self.id, options),
        );
    }

    fn resize_internal(&self, size: Size<Pixels>) {
        self.bounds.borrow_mut().size = size;
        let scale_factor = self.scale_factor.get();
        self.renderer.borrow_mut().update_drawable_size(Size {
            width: crate::DevicePixels((size.width.0 * scale_factor).round() as i32),
            height: crate::DevicePixels((size.height.0 * scale_factor).round() as i32),
        });
        let _ = self.event_tx.send(
            Priority::High,
            HeadlessEvent::Resized(self.id, size, scale_factor),
        );
    }
}

#[derive(Clone)]
struct HeadlessWindow(Rc<HeadlessWindowState>);

impl std::ops::Deref for HeadlessWindow {
    type Target = HeadlessWindowState;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl HasWindowHandle for HeadlessWindow {
    fn window_handle(
        &self,
    ) -> std::result::Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError>
    {
        unimplemented!("Headless windows are not backed by a native OS window")
    }
}

impl HasDisplayHandle for HeadlessWindow {
    fn display_handle(
        &self,
    ) -> std::result::Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError>
    {
        unimplemented!("Headless windows are not backed by a native display")
    }
}

impl PlatformWindow for HeadlessWindow {
    fn bounds(&self) -> Bounds<Pixels> {
        *self.bounds.borrow()
    }

    fn is_maximized(&self) -> bool {
        false
    }

    fn window_bounds(&self) -> WindowBounds {
        WindowBounds::Windowed(self.bounds())
    }

    fn content_size(&self) -> Size<Pixels> {
        self.bounds().size
    }

    fn resize(&mut self, size: Size<Pixels>) {
        self.resize_internal(size);
    }

    fn scale_factor(&self) -> f32 {
        self.scale_factor.get()
    }

    fn appearance(&self) -> WindowAppearance {
        WindowAppearance::Light
    }

    fn display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        Some(self.display.clone())
    }

    fn mouse_position(&self) -> Point<Pixels> {
        Point::default()
    }

    fn modifiers(&self) -> crate::Modifiers {
        crate::Modifiers::default()
    }

    fn capslock(&self) -> crate::Capslock {
        crate::Capslock::default()
    }

    fn set_input_handler(&mut self, input_handler: PlatformInputHandler) {
        self.input_handler.borrow_mut().replace(input_handler);
    }

    fn take_input_handler(&mut self) -> Option<PlatformInputHandler> {
        self.input_handler.borrow_mut().take()
    }

    fn prompt(
        &self,
        _level: crate::PromptLevel,
        _msg: &str,
        _detail: Option<&str>,
        _answers: &[PromptButton],
    ) -> Option<oneshot::Receiver<usize>> {
        None
    }

    fn activate(&self) {}

    fn is_active(&self) -> bool {
        self.active.get()
    }

    fn is_hovered(&self) -> bool {
        false
    }

    fn set_title(&mut self, title: &str) {
        self.title.borrow_mut().replace(title.to_string());
    }

    fn set_app_id(&mut self, app_id: &str) {
        self.app_id.borrow_mut().replace(app_id.to_string());
    }

    fn set_background_appearance(&self, background_appearance: WindowBackgroundAppearance) {
        self.background_appearance.set(background_appearance);
        self.renderer.borrow_mut().update_transparency(!matches!(
            background_appearance,
            WindowBackgroundAppearance::Opaque
        ));
    }

    fn minimize(&self) {}

    fn zoom(&self) {}

    fn toggle_fullscreen(&self) {}

    fn is_fullscreen(&self) -> bool {
        false
    }

    fn on_request_frame(&self, callback: Box<dyn FnMut(RequestFrameOptions)>) {
        *self.callbacks.on_request_frame.borrow_mut() = Some(callback);
    }

    fn on_input(
        &self,
        callback: Box<dyn FnMut(crate::PlatformInput) -> crate::DispatchEventResult>,
    ) {
        *self.callbacks.on_input.borrow_mut() = Some(callback);
    }

    fn on_active_status_change(&self, callback: Box<dyn FnMut(bool)>) {
        *self.callbacks.on_active_status_change.borrow_mut() = Some(callback);
    }

    fn on_hover_status_change(&self, callback: Box<dyn FnMut(bool)>) {
        *self.callbacks.on_hover_status_change.borrow_mut() = Some(callback);
    }

    fn on_resize(&self, callback: Box<dyn FnMut(Size<Pixels>, f32)>) {
        *self.callbacks.on_resize.borrow_mut() = Some(callback);
    }

    fn on_moved(&self, callback: Box<dyn FnMut()>) {
        *self.callbacks.on_moved.borrow_mut() = Some(callback);
    }

    fn on_should_close(&self, callback: Box<dyn FnMut() -> bool>) {
        *self.callbacks.on_should_close.borrow_mut() = Some(callback);
    }

    fn on_hit_test_window_control(&self, callback: Box<dyn FnMut() -> Option<WindowControlArea>>) {
        *self.callbacks.on_hit_test_window_control.borrow_mut() = Some(callback);
    }

    fn on_close(&self, callback: Box<dyn FnOnce()>) {
        *self.callbacks.on_close.borrow_mut() = Some(callback);
    }

    fn on_appearance_changed(&self, callback: Box<dyn FnMut()>) {
        *self.callbacks.on_appearance_changed.borrow_mut() = Some(callback);
    }

    fn request_frame(&self, options: RequestFrameOptions) {
        self.request_frame_internal(options);
    }

    fn draw(&self, scene: &crate::Scene) {
        self.renderer.borrow_mut().draw(scene);
    }

    fn present_framebuffer_only(&self) {}

    fn sprite_atlas(&self) -> Arc<dyn crate::PlatformAtlas> {
        self.sprite_atlas.clone()
    }

    fn close_programmatically(&self) {
        let _ = self
            .event_tx
            .send(Priority::High, HeadlessEvent::CloseWindow(self.id));
    }

    fn set_edited(&mut self, edited: bool) {
        self.edited.set(edited);
    }

    fn start_window_resize(&self, _edge: ResizeEdge) {}

    fn map_window(&mut self) -> anyhow::Result<()> {
        self.request_frame_internal(RequestFrameOptions {
            require_presentation: true,
            force_render: false,
        });
        Ok(())
    }

    fn gpu_specs(&self) -> Option<GpuSpecs> {
        None
    }

    fn update_ime_position(&self, _bounds: Bounds<Pixels>) {}

    fn create_wgpu_surface(
        &self,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Option<WgpuSurfaceHandle> {
        let registry = self.wgpu_context.surface_registry.clone();
        let surface_id = registry.create(self.wgpu_context.device(), width, height, format);
        let event_tx = self.event_tx.clone();
        let window_id = self.id;
        let present_trigger: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            let _ = event_tx.send(
                Priority::High,
                HeadlessEvent::RequestFrame(
                    window_id,
                    RequestFrameOptions {
                        require_presentation: true,
                        force_render: true,
                    },
                ),
            );
        });

        Some(WgpuSurfaceHandle::new(
            self.wgpu_context.device().clone(),
            self.wgpu_context.queue().clone(),
            surface_id,
            registry,
            present_trigger,
            None,
            width,
            height,
            format,
        ))
    }

    fn wgpu_output(&self) -> Option<WgpuOutputHandle> {
        Some(self.output.clone())
    }
}

pub(crate) struct HeadlessPlatform {
    background_executor: BackgroundExecutor,
    foreground_executor: ForegroundExecutor,
    dispatcher: Arc<HeadlessDispatcher>,
    main_tx: Arc<PriorityQueueSender<HeadlessEvent>>,
    main_rx: RefCell<PriorityQueueReceiver<HeadlessEvent>>,
    display: Rc<dyn PlatformDisplay>,
    text_system: Arc<dyn PlatformTextSystem>,
    active_cursor: Mutex<CursorStyle>,
    clipboard_item: Mutex<Option<ClipboardItem>>,
    windows: RefCell<HashMap<HeadlessWindowId, HeadlessWindow>>,
    active_window: RefCell<Option<AnyWindowHandle>>,
    next_window_id: Cell<u64>,
    quitting: AtomicBool,
    on_quit: RefCell<Option<Box<dyn FnMut()>>>,
    on_reopen: RefCell<Option<Box<dyn FnMut()>>>,
    on_app_menu_action: RefCell<Option<Box<dyn FnMut(&dyn crate::Action)>>>,
    on_will_open_app_menu: RefCell<Option<Box<dyn FnMut()>>>,
    on_validate_app_menu_command: RefCell<Option<Box<dyn FnMut(&dyn crate::Action) -> bool>>>,
    wgpu_context: RefCell<Option<Arc<WgpuContext>>>,
}

impl HeadlessPlatform {
    pub(crate) fn new() -> anyhow::Result<Rc<dyn Platform>> {
        let (main_tx, main_rx) = PriorityQueueReceiver::new();
        let main_tx = Arc::new(main_tx);
        let dispatcher = Arc::new(HeadlessDispatcher::new(main_tx.clone()));
        let background_executor = BackgroundExecutor::new(dispatcher.clone());
        let foreground_executor = ForegroundExecutor::new(dispatcher.clone());
        let display: Rc<dyn PlatformDisplay> = Rc::new(HeadlessDisplay::new());
        let text_system: Arc<dyn PlatformTextSystem> = Arc::new(CosmicTextSystem::new());

        Ok(Rc::new(Self {
            background_executor,
            foreground_executor,
            dispatcher,
            main_tx,
            main_rx: RefCell::new(main_rx),
            display,
            text_system,
            active_cursor: Mutex::new(CursorStyle::default()),
            clipboard_item: Mutex::new(None),
            windows: RefCell::new(HashMap::new()),
            active_window: RefCell::new(None),
            next_window_id: Cell::new(1),
            quitting: AtomicBool::new(false),
            on_quit: RefCell::new(None),
            on_reopen: RefCell::new(None),
            on_app_menu_action: RefCell::new(None),
            on_will_open_app_menu: RefCell::new(None),
            on_validate_app_menu_command: RefCell::new(None),
            wgpu_context: RefCell::new(None),
        }))
    }

    fn next_window_id(&self) -> HeadlessWindowId {
        let next_window_id = self.next_window_id.get();
        self.next_window_id.set(next_window_id + 1);
        HeadlessWindowId(next_window_id)
    }

    fn wgpu_context(&self) -> anyhow::Result<Arc<WgpuContext>> {
        if let Some(wgpu_context) = self.wgpu_context.borrow().clone() {
            return Ok(wgpu_context);
        }

        let wgpu_context = Arc::new(WgpuContext::new()?);
        self.wgpu_context.borrow_mut().replace(wgpu_context.clone());
        Ok(wgpu_context)
    }

    fn send_event(&self, priority: Priority, event: HeadlessEvent) {
        let _ = self.main_tx.send(priority, event);
    }

    fn run_main_loop(&self) {
        while !self.quitting.load(Ordering::Acquire) {
            let event = match self.main_rx.borrow_mut().pop() {
                Ok(event) => event,
                Err(_) => break,
            };

            match event {
                HeadlessEvent::Runnable(runnable) => match runnable {
                    RunnableVariant::Compat(runnable) => {
                        runnable.run();
                    }
                    RunnableVariant::Meta(runnable) => {
                        runnable.run();
                    }
                },
                HeadlessEvent::RequestFrame(window_id, options) => {
                    let window = self.windows.borrow().get(&window_id).cloned();
                    if let Some(window) = window {
                        HeadlessCallbacks::invoke_mut(
                            &window.callbacks.on_request_frame,
                            |callback| {
                                callback(options);
                            },
                        );
                    }
                }
                HeadlessEvent::Resized(window_id, size, scale_factor) => {
                    let window = self.windows.borrow().get(&window_id).cloned();
                    if let Some(window) = window {
                        HeadlessCallbacks::invoke_mut(&window.callbacks.on_resize, |callback| {
                            callback(size, scale_factor);
                        });
                    }
                }
                HeadlessEvent::CloseWindow(window_id) => {
                    let removed = self.windows.borrow_mut().remove(&window_id);
                    if let Some(window) = removed {
                        if self.active_window.borrow().as_ref() == Some(&window.handle) {
                            self.active_window.borrow_mut().take();
                        }
                        HeadlessCallbacks::invoke_once(&window.callbacks.on_close, |callback| {
                            callback();
                        });
                    }
                }
                HeadlessEvent::Quit => break,
            }
        }
    }
}

impl Platform for HeadlessPlatform {
    fn background_executor(&self) -> BackgroundExecutor {
        self.background_executor.clone()
    }

    fn foreground_executor(&self) -> ForegroundExecutor {
        self.foreground_executor.clone()
    }

    fn text_system(&self) -> Arc<dyn PlatformTextSystem> {
        self.text_system.clone()
    }

    fn run(&self, on_finish_launching: Box<dyn 'static + FnOnce()>) {
        on_finish_launching();
        if self.quitting.load(Ordering::Acquire) {
            return;
        }
        self.run_main_loop();
    }

    fn quit(&self) {
        if self.quitting.swap(true, Ordering::AcqRel) {
            return;
        }

        if let Some(mut callback) = self.on_quit.borrow_mut().take() {
            callback();
            self.on_quit.borrow_mut().replace(callback);
        }

        self.send_event(Priority::High, HeadlessEvent::Quit);
    }

    fn restart(&self, _binary_path: Option<PathBuf>) {}

    fn activate(&self, _ignoring_other_apps: bool) {}

    fn hide(&self) {}

    fn hide_other_apps(&self) {}

    fn unhide_other_apps(&self) {}

    fn displays(&self) -> Vec<Rc<dyn PlatformDisplay>> {
        vec![self.display.clone()]
    }

    fn primary_display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        Some(self.display.clone())
    }

    fn active_window(&self) -> Option<AnyWindowHandle> {
        *self.active_window.borrow()
    }

    fn open_window(
        &self,
        handle: AnyWindowHandle,
        params: WindowParams,
    ) -> anyhow::Result<Box<dyn PlatformWindow>> {
        let window_id = self.next_window_id();
        let window = HeadlessWindow(Rc::new(HeadlessWindowState::new(
            window_id,
            handle,
            params,
            self.display.clone(),
            self.wgpu_context()?,
            self.main_tx.clone(),
        )?));
        self.active_window.borrow_mut().replace(handle);
        self.windows.borrow_mut().insert(window_id, window.clone());
        Ok(Box::new(window))
    }

    fn window_appearance(&self) -> WindowAppearance {
        WindowAppearance::Light
    }

    fn open_url(&self, _url: &str) {}

    fn on_open_urls(&self, _callback: Box<dyn FnMut(Vec<String>)>) {}

    fn register_url_scheme(&self, _url: &str) -> Task<Result<()>> {
        Task::ready(Ok(()))
    }

    fn prompt_for_paths(
        &self,
        _options: crate::PathPromptOptions,
    ) -> oneshot::Receiver<Result<Option<Vec<PathBuf>>>> {
        let (tx, rx) = oneshot::channel();
        let _ = tx.send(Ok(None));
        rx
    }

    fn prompt_for_new_path(
        &self,
        _directory: &Path,
        _suggested_name: Option<&str>,
    ) -> oneshot::Receiver<Result<Option<PathBuf>>> {
        let (tx, rx) = oneshot::channel();
        let _ = tx.send(Ok(None));
        rx
    }

    fn can_select_mixed_files_and_dirs(&self) -> bool {
        true
    }

    fn reveal_path(&self, _path: &Path) {}

    fn open_with_system(&self, _path: &Path) {}

    fn on_quit(&self, callback: Box<dyn FnMut()>) {
        *self.on_quit.borrow_mut() = Some(callback);
    }

    fn on_reopen(&self, callback: Box<dyn FnMut()>) {
        *self.on_reopen.borrow_mut() = Some(callback);
    }

    fn enable_single_instance(&self, _app_id: &str) -> Result<()> {
        Ok(())
    }

    fn set_menus(&self, _menus: Vec<crate::Menu>, _keymap: &Keymap) {}

    fn set_dock_menu(&self, _menu: Vec<crate::MenuItem>, _keymap: &Keymap) {}

    fn on_app_menu_action(&self, callback: Box<dyn FnMut(&dyn crate::Action)>) {
        *self.on_app_menu_action.borrow_mut() = Some(callback);
    }

    fn on_will_open_app_menu(&self, callback: Box<dyn FnMut()>) {
        *self.on_will_open_app_menu.borrow_mut() = Some(callback);
    }

    fn on_validate_app_menu_command(&self, callback: Box<dyn FnMut(&dyn crate::Action) -> bool>) {
        *self.on_validate_app_menu_command.borrow_mut() = Some(callback);
    }

    fn app_path(&self) -> Result<PathBuf> {
        Ok(std::env::current_exe()?)
    }

    fn path_for_auxiliary_executable(&self, _name: &str) -> Result<PathBuf> {
        Err(anyhow!(
            "path_for_auxiliary_executable is not implemented for the headless platform"
        ))
    }

    fn set_cursor_style(&self, style: CursorStyle) {
        *self.active_cursor.lock() = style;
    }

    fn should_auto_hide_scrollbars(&self) -> bool {
        false
    }

    fn write_to_clipboard(&self, item: ClipboardItem) {
        *self.clipboard_item.lock() = Some(item);
    }

    fn read_from_clipboard(&self) -> Option<ClipboardItem> {
        self.clipboard_item.lock().clone()
    }

    fn write_credentials(&self, _url: &str, _username: &str, _password: &[u8]) -> Task<Result<()>> {
        Task::ready(Ok(()))
    }

    fn read_credentials(&self, _url: &str) -> Task<Result<Option<(String, Vec<u8>)>>> {
        Task::ready(Ok(None))
    }

    fn delete_credentials(&self, _url: &str) -> Task<Result<()>> {
        Task::ready(Ok(()))
    }

    fn keyboard_layout(&self) -> Box<dyn PlatformKeyboardLayout> {
        Box::new(HeadlessKeyboardLayout)
    }

    fn keyboard_mapper(&self) -> Rc<dyn PlatformKeyboardMapper> {
        Rc::new(DummyKeyboardMapper)
    }

    fn on_keyboard_layout_change(&self, _callback: Box<dyn FnMut()>) {}
}

struct HeadlessKeyboardLayout;

impl PlatformKeyboardLayout for HeadlessKeyboardLayout {
    fn id(&self) -> &str {
        "gpui.headless.keyboard"
    }

    fn name(&self) -> &str {
        "gpui.headless.keyboard"
    }
}
