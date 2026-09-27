#![allow(dead_code)]
//! A `GlobalAlloc` that stamps every block with its binary's tag, so a block
//! freed by a different binary's allocator than the one that allocated it (a
//! cross-DLL free) is always detected.
//!
//! The host test binary and the plugin fixture DLL each install one with a
//! different tag. Both use the same block layout on top of `System`, so a
//! cross free is physically valid; what happens to it depends on the mode:
//!
//! - strict (the default) aborts the process, for tests where no Rust value
//!   may cross the boundary at all;
//! - ledger mode frees the block and records it, so the combined heap of all
//!   binaries can still be balanced exactly (see [`combined_live_bytes`]).

use std::{
    alloc::{GlobalAlloc, Layout, System},
    io::Write,
    backtrace::Backtrace,
    collections::HashMap,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering},
    },
};

pub const HOST_TAG: u64 = u64::from_le_bytes(*b"HOSTHEAP");
pub const PLUGIN_TAG: u64 = u64::from_le_bytes(*b"PLUGHEAP");
const FREED_TAG: u64 = u64::from_le_bytes(*b"FREEDBLK");

/// Tag and size live in the 16 bytes directly before each block.
const HEADER: usize = 16;

pub struct TaggedAllocator {
    tag: u64,
    strict: AtomicBool,
    live_bytes: AtomicI64,
    live_blocks: AtomicI64,
    foreign_freed_bytes: AtomicI64,
    foreign_freed_blocks: AtomicI64,
    capturing_backtrace: AtomicBool,
    first_foreign_free: Mutex<Option<String>>,
    /// Physically live blocks by size (see [`bucket`]): incremented on every
    /// allocation, decremented on every free this allocator performs,
    /// including foreign ones, so summing all allocators' histograms gives
    /// the true live set.
    histogram: [AtomicI64; HISTOGRAM_LEN],
    /// Diagnostics: while `watch_recording` is set, every allocation of
    /// exactly `watch_size` bytes records its backtrace until it is freed
    /// (see [`TaggedAllocator::watch`]).
    watch_size: AtomicUsize,
    watch_recording: AtomicBool,
    in_watch: AtomicBool,
    watched: Mutex<Option<HashMap<usize, Backtrace>>>,
}

const EXACT_SIZES: usize = 4096;
pub const HISTOGRAM_LEN: usize = EXACT_SIZES + 52;

/// Sizes below 4096 bytes are counted exactly; larger ones by power of two.
fn bucket(size: usize) -> usize {
    if size < EXACT_SIZES {
        size
    } else {
        (EXACT_SIZES + (usize::BITS - 1 - size.leading_zeros()) as usize - 12).min(HISTOGRAM_LEN - 1)
    }
}

/// A readable label for a histogram index.
pub fn bucket_label(index: usize) -> String {
    if index < EXACT_SIZES {
        format!("{index} B")
    } else {
        let power = index - EXACT_SIZES + 12;
        format!("{}..{} B", 1u64 << power, 1u64 << (power + 1))
    }
}

/// A consistent reading of one allocator's ledger.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Ledger {
    /// Bytes this allocator allocated minus bytes of its own blocks it freed.
    pub live_bytes: i64,
    pub live_blocks: i64,
    /// Bytes of *other* allocators' blocks that this allocator freed.
    pub foreign_freed_bytes: i64,
    pub foreign_freed_blocks: i64,
}

/// Bytes physically live across every allocator whose ledger is given:
/// each cross free shows up as live in the allocating ledger and as foreign
/// in the freeing one, so subtracting the foreign frees balances them.
pub fn combined_live_bytes(ledgers: &[Ledger]) -> i64 {
    ledgers
        .iter()
        .map(|ledger| ledger.live_bytes - ledger.foreign_freed_bytes)
        .sum()
}

pub fn combined_live_blocks(ledgers: &[Ledger]) -> i64 {
    ledgers
        .iter()
        .map(|ledger| ledger.live_blocks - ledger.foreign_freed_blocks)
        .sum()
}

impl TaggedAllocator {
    pub const fn new(tag: u64) -> Self {
        Self {
            tag,
            strict: AtomicBool::new(true),
            live_bytes: AtomicI64::new(0),
            live_blocks: AtomicI64::new(0),
            foreign_freed_bytes: AtomicI64::new(0),
            foreign_freed_blocks: AtomicI64::new(0),
            capturing_backtrace: AtomicBool::new(false),
            first_foreign_free: Mutex::new(None),
            histogram: [const { AtomicI64::new(0) }; HISTOGRAM_LEN],
            watch_size: AtomicUsize::new(0),
            watch_recording: AtomicBool::new(false),
            in_watch: AtomicBool::new(false),
            watched: Mutex::new(None),
        }
    }

    pub fn set_strict(&self, strict: bool) {
        self.strict.store(strict, Ordering::SeqCst);
    }

    pub fn live_bytes(&self) -> i64 {
        self.live_bytes.load(Ordering::SeqCst)
    }

    pub fn live_blocks(&self) -> i64 {
        self.live_blocks.load(Ordering::SeqCst)
    }

    /// Copies the live-blocks-by-size histogram into `out`.
    pub fn size_histogram(&self, out: &mut [i64; HISTOGRAM_LEN]) {
        for (count, slot) in out.iter_mut().zip(&self.histogram) {
            *count = slot.load(Ordering::SeqCst);
        }
    }

    pub fn ledger(&self) -> Ledger {
        Ledger {
            live_bytes: self.live_bytes.load(Ordering::SeqCst),
            live_blocks: self.live_blocks.load(Ordering::SeqCst),
            foreign_freed_bytes: self.foreign_freed_bytes.load(Ordering::SeqCst),
            foreign_freed_blocks: self.foreign_freed_blocks.load(Ordering::SeqCst),
        }
    }

    /// The backtrace of the first cross free this allocator performed.
    pub fn first_foreign_free(&self) -> Option<String> {
        self.first_foreign_free
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
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

    fn record_foreign_free(&self, size: usize) {
        self.foreign_freed_bytes
            .fetch_add(size as i64, Ordering::SeqCst);
        self.foreign_freed_blocks.fetch_add(1, Ordering::SeqCst);
        // Capturing allocates, which re-enters this allocator; the flag keeps
        // that from recursing into another capture.
        if self
            .capturing_backtrace
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            let mut first = self
                .first_foreign_free
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if first.is_none() {
                *first = Some(std::backtrace::Backtrace::force_capture().to_string());
            }
            drop(first);
            // Deliberately left set once captured, so later cross frees stay cheap.
        }
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
            self.histogram[bucket(layout.size())].fetch_add(1, Ordering::SeqCst);
            if self.watch_recording.load(Ordering::Relaxed)
                && layout.size() == self.watch_size.load(Ordering::Relaxed)
            {
                self.watch_insert(block as usize);
            }
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
            if tag != HOST_TAG && tag != PLUGIN_TAG {
                Self::fail("tagged allocator: freed a block no tagged allocator produced\n");
            }
            if block.sub(8).cast::<u64>().read() != layout.size() as u64 {
                Self::fail("tagged allocator: block freed with a different size than it was allocated with\n");
            }
            if tag == self.tag {
                self.live_bytes
                    .fetch_sub(layout.size() as i64, Ordering::SeqCst);
                self.live_blocks.fetch_sub(1, Ordering::SeqCst);
            } else if self.strict.load(Ordering::SeqCst) {
                Self::fail(
                    "tagged allocator: block freed by a different binary's allocator than the one \
                     that allocated it (cross-DLL free)\n",
                );
            } else {
                self.record_foreign_free(layout.size());
            }
            if layout.size() == self.watch_size.load(Ordering::Relaxed) {
                self.watch_remove(block as usize);
            }
            self.histogram[bucket(layout.size())].fetch_sub(1, Ordering::SeqCst);
            block.sub(16).cast::<u64>().write(FREED_TAG);
            System.dealloc(block.sub(header), outer);
        }
    }
}

impl TaggedAllocator {
    /// Start watching allocations of exactly `size` bytes: while recording
    /// (see [`Self::set_watch_recording`]), each one keeps its backtrace
    /// until it is freed, so the survivors of a phase can be attributed.
    pub fn watch(&self, size: usize) {
        self.in_watch.store(true, Ordering::SeqCst);
        *self.watched.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(HashMap::new());
        self.in_watch.store(false, Ordering::SeqCst);
        self.watch_size.store(size, Ordering::SeqCst);
    }

    pub fn set_watch_recording(&self, recording: bool) {
        self.watch_recording.store(recording, Ordering::SeqCst);
    }

    /// Watched blocks still alive, grouped by where they were allocated
    /// (the first gpui frames of their backtrace), most numerous first.
    pub fn watch_report(&self, frames: usize) -> Vec<(usize, String)> {
        self.in_watch.store(true, Ordering::SeqCst);
        let mut groups: HashMap<String, usize> = HashMap::new();
        if let Some(watched) = self
            .watched
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            for backtrace in watched.values() {
                let rendered = backtrace.to_string();
                let signature: Vec<&str> = rendered
                    .lines()
                    .filter(|line| !line.trim_start().starts_with("at "))
                    .skip_while(|line| !line.contains("gpui::"))
                    .take(frames)
                    .collect();
                *groups.entry(signature.join("\n")).or_insert(0) += 1;
            }
        }
        let mut report: Vec<(usize, String)> = groups
            .into_iter()
            .map(|(signature, count)| (count, signature))
            .collect();
        report.sort_by(|a, b| b.0.cmp(&a.0));
        self.in_watch.store(false, Ordering::SeqCst);
        report
    }

    fn watch_insert(&self, address: usize) {
        if self.in_watch.swap(true, Ordering::SeqCst) {
            return;
        }
        if let Some(watched) = self
            .watched
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_mut()
        {
            watched.insert(address, Backtrace::force_capture());
        }
        self.in_watch.store(false, Ordering::SeqCst);
    }

    fn watch_remove(&self, address: usize) {
        if self.in_watch.swap(true, Ordering::SeqCst) {
            return;
        }
        if let Some(watched) = self
            .watched
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_mut()
        {
            watched.remove(&address);
        }
        self.in_watch.store(false, Ordering::SeqCst);
    }
}
