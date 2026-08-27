//! One primitive kind's arena, as a real `wgpu::Buffer`.
//! See docs/gpu-native-architecture.md §3.5, §5.0.
//!
//! Phase 1 produced §5.0's upload *instructions* — a list of
//! `wgpui_core::scene::UploadRange`s naming exactly the bytes that changed —
//! and `docs/phase-1-results.md` recorded that no real buffer existed to apply
//! them to, "correctly deferred to `wgpui-wgpu`". This is the smallest thing
//! that applies them.
//!
//! # What this is not
//!
//! It is not `slab_gpu.rs`'s successor. That file is 1,437 lines of CPU-side
//! allocator, and its successor is `wgpui_core::scene::slab`, which already
//! exists and already decides every placement. What is left over — "hold a
//! buffer, grow it when the arena grows, and turn an `UploadRange` into a
//! `write_buffer`" — is this, and it is deliberately about that much code.
//!
//! Delta-upload *coalescing* is likewise already done, in
//! `wgpui_core::scene::slab_range::coalesce_uploads`, before the instructions
//! reach here. Repeating it on this side would be a second implementation of a
//! decision the protocol already made.

use wgpui_core::scene::UploadRange;

/// A kind's arena buffer, grown by high-water mark.
pub struct SlabBuffer {
    buffer: wgpu::Buffer,
    capacity: u64,
    label: &'static str,
    allocations: u64,
    uploaded_bytes: u64,
    upload_calls: u64,
}

impl SlabBuffer {
    /// An empty arena.
    pub fn new(device: &wgpu::Device, label: &'static str) -> SlabBuffer {
        let capacity = 4096;
        SlabBuffer {
            buffer: create(device, label, capacity),
            capacity,
            label,
            allocations: 1,
            uploaded_bytes: 0,
            upload_calls: 0,
        }
    }

    /// The buffer, for binding.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Bytes the buffer can hold.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// How many times the buffer has been (re)allocated. A steady-state frame
    /// loop must not keep raising this.
    pub fn allocations(&self) -> u64 {
        self.allocations
    }

    /// Bytes written since creation.
    pub fn uploaded_bytes(&self) -> u64 {
        self.uploaded_bytes
    }

    /// `write_buffer` calls made since creation.
    pub fn upload_calls(&self) -> u64 {
        self.upload_calls
    }

    /// Make room for `bytes`, reporting whether the buffer was replaced.
    ///
    /// A replaced buffer invalidates every bind group naming it, so the caller
    /// has to know rather than find out by rendering nothing. Doubling rather
    /// than exact sizing, for the reason every growable buffer does it: a layer
    /// that gains one primitive must not force a reallocation.
    pub fn reserve(&mut self, device: &wgpu::Device, bytes: u64) -> bool {
        if self.capacity >= bytes {
            return false;
        }
        let capacity = bytes.max(self.capacity.saturating_mul(2)).max(4096);
        self.buffer = create(device, self.label, capacity);
        self.capacity = capacity;
        self.allocations += 1;
        true
    }

    /// Apply §5.0's upload instructions against `resident`.
    ///
    /// `resident` is the kind's whole CPU-side arena
    /// (`PrimitiveStore::resident_bytes`); each range names a span of it. A
    /// range that runs past the resident buffer is skipped rather than
    /// truncated — it means the instruction list and the store disagree, which
    /// is a bookkeeping bug the caller should see as a wrong picture rather
    /// than as a device error that aborts the process.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resident: &[u8],
        ranges: &[UploadRange],
    ) {
        self.reserve(device, resident.len() as u64);
        for range in ranges {
            let start = usize::try_from(range.byte_offset).unwrap_or(usize::MAX);
            let end = usize::try_from(range.byte_end()).unwrap_or(usize::MAX);
            let Some(bytes) = resident.get(start..end) else {
                continue;
            };
            if bytes.is_empty() {
                continue;
            }
            queue.write_buffer(&self.buffer, range.byte_offset, bytes);
            self.uploaded_bytes += bytes.len() as u64;
            self.upload_calls += 1;
        }
    }

    /// Write the whole arena, for a first frame or a reallocation.
    pub fn upload_all(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, resident: &[u8]) {
        self.reserve(device, resident.len() as u64);
        if resident.is_empty() {
            return;
        }
        queue.write_buffer(&self.buffer, 0, resident);
        self.uploaded_bytes += resident.len() as u64;
        self.upload_calls += 1;
    }
}

fn create(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
