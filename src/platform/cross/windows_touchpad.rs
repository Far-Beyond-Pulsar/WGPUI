//! Precision touchpad scrolling on Windows, through DirectManipulation.
//!
//! Without it a touchpad reaches the window as emulated wheel messages, and a
//! fast flick arrives as a few jumps of many lines with long pauses between
//! them. DirectManipulation reports the fingers' movement and the coast after
//! they lift as a stream of small pixel steps.
//!
//! The window offers it every touchpad contact (`DM_POINTERHITTEST`); the
//! mouse wheel is untouched. The viewport is updated by hand from the event
//! loop while a gesture or its coast is running, and its content transform is
//! turned into scroll steps. `WGPUI_DIRECT_MANIPULATION=0` turns this off.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::DirectManipulation::*;
use windows::Win32::System::Com::{CLSCTX_INPROC_SERVER, CoCreateInstance};
use windows::Win32::UI::Input::Pointer::GetPointerType;
use windows::Win32::UI::Shell::{DefSubclassProc, RemoveWindowSubclass, SetWindowSubclass};
use windows::Win32::UI::WindowsAndMessaging::{POINTER_INPUT_TYPE, PT_TOUCHPAD};
use windows::core::{Ref, Result, implement};

use crate::TouchPhase;

/// Sent when a touchpad contact lands, so a window can claim it.
const DM_POINTERHITTEST: u32 = 0x0250;
const SUBCLASS_ID: usize = 0x5747_5055;
/// The viewport's size. Only the content's movement within it is read, so any
/// size does.
const VIEWPORT: i32 = 1000;
/// Wheel lines a pinch that doubles the scale stands for.
const PINCH_LINES_PER_DOUBLING: f32 = 10.0;

/// One step of a gesture, in physical pixels.
pub(crate) enum TouchpadStep {
    Scroll {
        x: f32,
        y: f32,
        phase: TouchPhase,
    },
    /// A pinch, as the Ctrl+wheel lines Windows would have sent for it.
    Zoom {
        lines: f32,
    },
}

#[derive(Default)]
struct Gesture {
    /// A contact was claimed and the gesture, coast or reset is not over.
    active: Cell<bool>,
    /// Scroll steps have been reported and not yet ended.
    scrolling: Cell<bool>,
    /// The content is being put back to its origin; that is not input.
    resetting: Cell<bool>,
    /// The content's scale and offset at the last update.
    last: Cell<(f32, f32, f32)>,
    steps: RefCell<Vec<TouchpadStep>>,
}

#[implement(IDirectManipulationViewportEventHandler)]
struct Handler {
    gesture: Rc<Gesture>,
}

impl IDirectManipulationViewportEventHandler_Impl for Handler_Impl {
    fn OnViewportStatusChanged(
        &self,
        viewport: Ref<'_, IDirectManipulationViewport>,
        current: DIRECTMANIPULATION_STATUS,
        previous: DIRECTMANIPULATION_STATUS,
    ) -> Result<()> {
        if current != DIRECTMANIPULATION_READY || previous == DIRECTMANIPULATION_READY {
            return Ok(());
        }
        let gesture = &self.gesture;
        if gesture.scrolling.take() {
            gesture.steps.borrow_mut().push(TouchpadStep::Scroll {
                x: 0.0,
                y: 0.0,
                phase: TouchPhase::Ended,
            });
        }
        let at_origin = gesture.last.get() == (1.0, 0.0, 0.0);
        if gesture.resetting.take() || at_origin {
            gesture.active.set(false);
        } else if let Some(viewport) = viewport.as_ref() {
            // Start the next gesture from the origin, so the content never
            // reaches the edge of what can be panned.
            gesture.resetting.set(true);
            // SAFETY: a live viewport, on the thread that made it.
            unsafe { viewport.ZoomToRect(0.0, 0.0, VIEWPORT as f32, VIEWPORT as f32, false)? };
        }
        Ok(())
    }

    fn OnViewportUpdated(&self, _: Ref<'_, IDirectManipulationViewport>) -> Result<()> {
        Ok(())
    }

    fn OnContentUpdated(
        &self,
        _: Ref<'_, IDirectManipulationViewport>,
        content: Ref<'_, IDirectManipulationContent>,
    ) -> Result<()> {
        let Some(content) = content.as_ref() else {
            return Ok(());
        };
        let mut matrix = [0.0f32; 6];
        // SAFETY: a live content object and a matrix of the size it fills.
        unsafe { content.GetContentTransform(&mut matrix)? };
        let (scale, x, y) = (matrix[0], matrix[4], matrix[5]);
        let gesture = &self.gesture;
        let (last_scale, last_x, last_y) = gesture.last.replace((scale, x, y));
        if gesture.resetting.get() || scale <= 0.0 {
            return Ok(());
        }
        let ratio = scale / last_scale;
        if (ratio - 1.0).abs() > 1e-4 {
            // While pinching, the offset moves to keep the pinch centred.
            gesture.steps.borrow_mut().push(TouchpadStep::Zoom {
                lines: ratio.log2() * PINCH_LINES_PER_DOUBLING,
            });
            return Ok(());
        }
        let (x, y) = (x - last_x, y - last_y);
        if x != 0.0 || y != 0.0 {
            let phase = if gesture.scrolling.replace(true) {
                TouchPhase::Moved
            } else {
                TouchPhase::Started
            };
            gesture
                .steps
                .borrow_mut()
                .push(TouchpadStep::Scroll { x, y, phase });
        }
        Ok(())
    }
}

/// What the window procedure hook needs.
struct Hook {
    viewport: IDirectManipulationViewport,
    gesture: Rc<Gesture>,
}

unsafe extern "system" fn window_hook(
    window: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
    _id: usize,
    data: usize,
) -> LRESULT {
    if message == DM_POINTERHITTEST {
        // SAFETY: `data` is the boxed `Hook` its `Touchpad` keeps until after
        // it removes this procedure.
        let hook = unsafe { &*(data as *const Hook) };
        let pointer = (wparam.0 & 0xffff) as u32;
        let mut kind = POINTER_INPUT_TYPE::default();
        // SAFETY: `kind` receives the type of the pointer in this message.
        if unsafe { GetPointerType(pointer, &mut kind) }.is_ok() && kind == PT_TOUCHPAD {
            // A coast being stopped by this contact must not look like a reset.
            hook.gesture.resetting.set(false);
            // SAFETY: a live viewport, on the thread that made it.
            match unsafe { hook.viewport.SetContact(pointer) } {
                Ok(()) => hook.gesture.active.set(true),
                Err(error) => log::warn!("Touchpad contact not claimed: {error}"),
            }
        }
    }
    // SAFETY: passes the message on, as every subclass procedure must.
    unsafe { DefSubclassProc(window, message, wparam, lparam) }
}

/// DirectManipulation for one window. It must be made, used and dropped on the
/// window's thread, while the window exists.
pub(crate) struct Touchpad {
    window: HWND,
    manager: IDirectManipulationManager,
    updates: IDirectManipulationUpdateManager,
    viewport: IDirectManipulationViewport,
    handler: u32,
    hook: Box<Hook>,
}

impl Touchpad {
    /// COM must already be initialised on this thread (winit does so).
    pub(crate) fn new(window: HWND) -> Result<Self> {
        let gesture = Rc::new(Gesture::default());
        gesture.last.set((1.0, 0.0, 0.0));
        // SAFETY: plain DirectManipulation setup for a window of this thread,
        // in the documented order.
        unsafe {
            let manager: IDirectManipulationManager =
                CoCreateInstance(&DirectManipulationManager, None, CLSCTX_INPROC_SERVER)?;
            let updates: IDirectManipulationUpdateManager = manager.GetUpdateManager()?;
            let viewport: IDirectManipulationViewport = manager.CreateViewport(None, window)?;
            viewport.ActivateConfiguration(
                DIRECTMANIPULATION_CONFIGURATION_INTERACTION
                    | DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_X
                    | DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_Y
                    | DIRECTMANIPULATION_CONFIGURATION_TRANSLATION_INERTIA
                    | DIRECTMANIPULATION_CONFIGURATION_SCALING,
            )?;
            viewport.SetViewportOptions(DIRECTMANIPULATION_VIEWPORT_OPTIONS_MANUALUPDATE)?;
            let handler: IDirectManipulationViewportEventHandler = Handler {
                gesture: gesture.clone(),
            }
            .into();
            let handler = viewport.AddEventHandler(Some(window), &handler)?;
            viewport.SetViewportRect(&RECT {
                left: 0,
                top: 0,
                right: VIEWPORT,
                bottom: VIEWPORT,
            })?;
            manager.Activate(window)?;
            viewport.Enable()?;
            updates.Update(None)?;

            let hook = Box::new(Hook {
                viewport: viewport.clone(),
                gesture,
            });
            let data = std::ptr::from_ref::<Hook>(&hook) as usize;
            SetWindowSubclass(window, Some(window_hook), SUBCLASS_ID, data).ok()?;
            Ok(Self {
                window,
                manager,
                updates,
                viewport,
                handler,
                hook,
            })
        }
    }

    /// Advance a running gesture and take the steps it produced. Does nothing
    /// while the touchpad is idle.
    pub(crate) fn poll(&self) -> Vec<TouchpadStep> {
        let gesture = &self.hook.gesture;
        if gesture.active.get() {
            // SAFETY: a live update manager, on the thread that made it. The
            // handler runs inside this call.
            if let Err(error) = unsafe { self.updates.Update(None) } {
                log::warn!("Touchpad update failed: {error}");
            }
        }
        std::mem::take(&mut *gesture.steps.borrow_mut())
    }
}

impl Drop for Touchpad {
    fn drop(&mut self) {
        // SAFETY: undoes `new` on the same thread. The hook goes first, so
        // nothing reaches `self.hook` afterwards. Failures leave nothing to do
        // for a window that is going away.
        unsafe {
            let _ = RemoveWindowSubclass(self.window, Some(window_hook), SUBCLASS_ID);
            let _ = self.viewport.Stop();
            let _ = self.viewport.RemoveEventHandler(self.handler);
            let _ = self.viewport.Abandon();
            let _ = self.manager.Deactivate(self.window);
        }
    }
}
