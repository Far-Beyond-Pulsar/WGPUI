//! A `GlobalAlloc` that stamps every block with its binary's tag and aborts
//! the process if a block is freed by a different binary's allocator.
//!
//! The host test binary and the plugin fixture DLL each install one with a
//! different tag, so a cross-binary free (memory allocated by one copy of
//! gpui and freed by another, the bug class the shared runtime exists to
//! prevent) kills the test on the spot instead of silently corrupting a heap
//! or skewing a tracking allocator's numbers.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    io::Write,
    sync::atomic::{AtomicI64, Ordering},
};

pub const HOST_TAG: u64 = u64::from_le_bytes(*b"HOSTHEAP");
pub const PLUGIN_TAG: u64 = u64::from_le_bytes(*b"PLUGHEAP");
const FREED_TAG: u64 = u64::from_le_bytes(*b"FREEDBLK");

/// Tag and size live in the 16 bytes directly before each block.
const HEADER: usize = 16;

pub struct TaggedAllocator {
    tag: u64,
    live_bytes: AtomicI64,
    live_blocks: AtomicI64,
}

impl TaggedAllocator {
    pub const fn new(tag: u64) -> Self {
        Self {
            tag,
            live_bytes: AtomicI64::new(0),
            live_blocks: AtomicI64::new(0),
        }
    }

    pub fn live_bytes(&self) -> i64 {
        self.live_bytes.load(Ordering::SeqCst)
    }

    pub fn live_blocks(&self) -> i64 {
        self.live_blocks.load(Ordering::SeqCst)
    }

    fn outer_layout(layout: Layout) -> Option<(Layout, usize)> {
        let header = layout.align().max(HEADER);
        let size = layout.size().checked_add(header)?;
        Some((Layout::from_size_align(size, header).ok()?, header))
    }

    fn fail(message: &str) -> ! {
        let mut stderr = std::io::stderr();
        // Nothing useful can be done if stderr is gone; abort regardless.
        if stderr.write_all(message.as_bytes()).is_err() {
            std::process::abort();
        }
        std::process::abort();
    }
}

unsafe impl GlobalAlloc for TaggedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let Some((outer, header)) = Self::outer_layout(layout) else {
            return std::ptr::null_mut();
        };
        unsafe {
            let base = System.alloc(outer);
            if base.is_null() {
                return base;
            }
            let block = base.add(header);
            block.sub(16).cast::<u64>().write(self.tag);
            block.sub(8).cast::<u64>().write(layout.size() as u64);
            self.live_bytes
                .fetch_add(layout.size() as i64, Ordering::SeqCst);
            self.live_blocks.fetch_add(1, Ordering::SeqCst);
            block
        }
    }

    unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
        let Some((outer, header)) = Self::outer_layout(layout) else {
            Self::fail("tagged allocator: dealloc with an impossible layout\n");
        };
        unsafe {
            let tag = block.sub(16).cast::<u64>().read();
            if tag == FREED_TAG {
                Self::fail("tagged allocator: double free\n");
            }
            if tag != self.tag {
                Self::fail(
                    "tagged allocator: block freed by a different binary's allocator than the one \
                     that allocated it (cross-DLL free)\n",
                );
            }
            if block.sub(8).cast::<u64>().read() != layout.size() as u64 {
                Self::fail("tagged allocator: block freed with a different size than it was allocated with\n");
            }
            block.sub(16).cast::<u64>().write(FREED_TAG);
            self.live_bytes
                .fetch_sub(layout.size() as i64, Ordering::SeqCst);
            self.live_blocks.fetch_sub(1, Ordering::SeqCst);
            System.dealloc(block.sub(header), outer);
        }
    }
}
