use anyhow::{Context as _, Result};
use objc2::{
    ClassType, DeclaredClass, declare_class, msg_send, msg_send_id, mutability, rc::Retained, sel,
};
use objc2_foundation::{
    MainThreadMarker, NSAppleEventDescriptor, NSAppleEventManager, NSObject, NSObjectProtocol,
};
use winit::event_loop::EventLoopProxy;

use super::dispatcher::CrossEvent;

const CORE_EVENT_CLASS: u32 = u32::from_be_bytes(*b"aevt");
const OPEN_DOCUMENTS: u32 = u32::from_be_bytes(*b"odoc");
const GET_URL: u32 = u32::from_be_bytes(*b"GURL");
const DIRECT_OBJECT: u32 = u32::from_be_bytes(*b"----");

declare_class!(
    struct OpenUrlsReceiver;

    unsafe impl ClassType for OpenUrlsReceiver {
        type Super = NSObject;
        type Mutability = mutability::MainThreadOnly;
        const NAME: &'static str = "WgpuiOpenUrlsReceiver";
    }

    impl DeclaredClass for OpenUrlsReceiver {
        type Ivars = EventLoopProxy<CrossEvent>;
    }

    unsafe impl NSObjectProtocol for OpenUrlsReceiver {}

    unsafe impl OpenUrlsReceiver {
        #[method(handleOpenDocuments:withReplyEvent:)]
        fn handle_open_documents(&self, event: &NSAppleEventDescriptor, _reply: &NSAppleEventDescriptor) {
            // AppKit delivers this list before launch as well as while the app is running.
            // Queue it through winit so application code never runs inside an Objective-C callback.
            let urls = unsafe {
                let documents: Option<Retained<NSAppleEventDescriptor>> =
                    msg_send_id![event, paramDescriptorForKeyword: DIRECT_OBJECT];
                let Some(documents) = documents else { return };
                (1..=documents.numberOfItems())
                    .filter_map(|index| {
                        let document = documents.descriptorAtIndex(index)?;
                        let url = document.fileURLValue()?;
                        Some(url.absoluteString()?.to_string())
                    })
                    .collect()
            };
            self.send_urls(urls);
        }

        #[method(handleGetUrl:withReplyEvent:)]
        fn handle_get_url(&self, event: &NSAppleEventDescriptor, _reply: &NSAppleEventDescriptor) {
            // These selectors use the documented NSAppleEventDescriptor and OSType signatures.
            let url = unsafe {
                let descriptor: Option<Retained<NSAppleEventDescriptor>> =
                    msg_send_id![event, paramDescriptorForKeyword: DIRECT_OBJECT];
                descriptor.and_then(|descriptor| descriptor.stringValue()).map(|url| url.to_string())
            };
            if let Some(url) = url {
                self.send_urls(vec![url]);
            }
        }
    }
);

impl OpenUrlsReceiver {
    fn send_urls(&self, urls: Vec<String>) {
        if !urls.is_empty()
            && let Err(error) = self.ivars().send_event(CrossEvent::OpenUrls(urls))
        {
            log::error!("failed to queue macOS open URLs event: {error}");
        }
    }
}

pub(super) struct OpenUrlsHandler {
    _receiver: Retained<OpenUrlsReceiver>,
}

impl OpenUrlsHandler {
    pub(super) fn new(proxy: EventLoopProxy<CrossEvent>) -> Result<Self> {
        let main_thread =
            MainThreadMarker::new().context("macOS open URLs handler requires the main thread")?;
        let receiver = main_thread.alloc::<OpenUrlsReceiver>().set_ivars(proxy);
        // SAFETY: NSObject initialization and Apple event registration run on the main thread.
        // The receiver implements both selectors with the signatures required by NSAppleEventManager.
        let receiver: Retained<OpenUrlsReceiver> = unsafe {
            let receiver: Retained<OpenUrlsReceiver> = msg_send_id![super(receiver), init];
            let manager = NSAppleEventManager::sharedAppleEventManager();
            let (): () = msg_send![&*manager,
                setEventHandler: &*receiver
                andSelector: sel!(handleOpenDocuments:withReplyEvent:)
                forEventClass: CORE_EVENT_CLASS
                andEventID: OPEN_DOCUMENTS
            ];
            let (): () = msg_send![&*manager,
                setEventHandler: &*receiver
                andSelector: sel!(handleGetUrl:withReplyEvent:)
                forEventClass: GET_URL
                andEventID: GET_URL
            ];
            receiver
        };
        Ok(Self {
            _receiver: receiver,
        })
    }
}

impl Drop for OpenUrlsHandler {
    fn drop(&mut self) {
        // SAFETY: CrossPlatform and its receiver are main-thread-only and outlive these registrations.
        unsafe {
            let manager = NSAppleEventManager::sharedAppleEventManager();
            let (): () = msg_send![&*manager, removeEventHandlerForEventClass: CORE_EVENT_CLASS andEventID: OPEN_DOCUMENTS];
            let (): () =
                msg_send![&*manager, removeEventHandlerForEventClass: GET_URL andEventID: GET_URL];
        }
    }
}
