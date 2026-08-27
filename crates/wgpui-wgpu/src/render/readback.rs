//! Reading GPU-written buffers back to the CPU.
//! See docs/gpu-native-architecture.md §5.3, §3.5.
//!
//! §3.5 names this file for the CPU-readback *fallback* path for indirect draw
//! — "compute writes the args, CPU reads them back and issues direct draws".
//! Phase 3 built the primitive underneath it ([`read_u32_buffer`]: map one
//! storage buffer's contents into a `Vec<u32>` synchronously) and said the
//! fallback itself was Phase 4's job.
//!
//! Phase 3's two callers, both honest about what they cost:
//!
//!  - The ordering pass reads a four-byte convergence counter to tell a fixed
//!    point from a truncated relaxation budget. That read is *inside* the
//!    measured window, because a pass that skipped it would not be correct.
//!  - The differential harness reads the whole result back to compare it
//!    against the CPU reference. That read is deliberately *outside* the
//!    measured window, for the reason Phase 0's Spike A gives: a real consumer
//!    would not round-trip its results through the CPU either.
//!
//! # What Phase 4 found this needed, and what it did not
//!
//! [`read_u32_buffer`] is reusable **as-is for correctness work** — indirect
//! argument records are four `u32`s each, so a differential test reads them
//! with no new code at all, and Phase 4's does.
//!
//! It is *not* reusable as-is for the fallback itself, for one specific
//! reason: it creates a staging buffer per call. Phase 3's callers run once per
//! measurement; the fallback runs **once per kind per frame, forever**, so
//! shipping it on this function would allocate and destroy GPU memory every
//! frame in the one path that exists because the device is already the weak
//! one. [`StagingReader`] is the extension — the same read, against a staging
//! buffer that is created once, grown by high-water mark, and reused. Nothing
//! about the mapping, the polling, or the error vocabulary changes; the
//! allocation does.

/// Why a readback failed.
#[derive(Debug)]
pub enum ReadbackError {
    /// The device could not be polled to completion.
    Poll(wgpu::PollError),
    /// The buffer could not be mapped.
    Map(wgpu::BufferAsyncError),
    /// The mapping callback's channel closed before delivering a result — the
    /// device was dropped or lost mid-map.
    Cancelled,
    /// The mapped range could not be viewed.
    Range(wgpu::MapRangeError),
}

impl std::fmt::Display for ReadbackError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadbackError::Poll(error) => write!(formatter, "device poll failed: {error}"),
            ReadbackError::Map(error) => write!(formatter, "buffer map failed: {error}"),
            ReadbackError::Cancelled => write!(formatter, "buffer map was cancelled"),
            ReadbackError::Range(error) => write!(formatter, "mapped range unavailable: {error}"),
        }
    }
}

impl std::error::Error for ReadbackError {}

/// Copy `count` `u32`s out of `source` and return them.
///
/// `source` must carry `BufferUsages::COPY_SRC`. Submits its own encoder and
/// blocks until the copy and the map complete, so the caller gets values rather
/// than a future — every Phase 3 consumer is a synchronous measurement or a
/// test.
pub fn read_u32_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<u32>, ReadbackError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let size = (count * std::mem::size_of::<u32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        // The receiver is alive until this function returns, so a send failure
        // means the device was torn down under us; the receive below reports it
        // as `Cancelled` rather than this closure having anywhere to raise it.
        if sender.send(result).is_err() {
            log_dropped_readback();
        }
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(ReadbackError::Poll)?;
    receiver
        .recv()
        .map_err(|_| ReadbackError::Cancelled)?
        .map_err(ReadbackError::Map)?;

    let values = {
        let view = slice.get_mapped_range().map_err(ReadbackError::Range)?;
        let mut values = Vec::with_capacity(count);
        for chunk in view.chunks_exact(4) {
            values.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        values
    };
    staging.unmap();
    Ok(values)
}

/// A reusable staging buffer for a readback that happens every frame.
///
/// Same read as [`read_u32_buffer`], same errors, one difference: the staging
/// buffer survives between calls and is only recreated when a call needs more
/// room than the last one did. See this module's doc for why the fallback path
/// specifically cannot use the per-call form.
#[derive(Debug, Default)]
pub struct StagingReader {
    staging: Option<wgpu::Buffer>,
    capacity: u64,
    /// How many times the staging buffer has been (re)allocated. A frame loop
    /// in steady state must not keep growing this — asserted by
    /// `render/draw.rs`'s fallback test rather than left as an intention.
    allocations: u64,
}

impl StagingReader {
    /// A reader holding no buffer yet.
    pub fn new() -> StagingReader {
        StagingReader::default()
    }

    /// How many times a staging buffer has been allocated.
    pub fn allocations(&self) -> u64 {
        self.allocations
    }

    /// Copy `count` `u32`s out of `source`, reusing this reader's staging
    /// buffer.
    ///
    /// `source` must carry `BufferUsages::COPY_SRC`. Appends into `destination`
    /// after clearing it, so a caller in a frame loop reuses its `Vec` too.
    pub fn read_u32(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::Buffer,
        count: usize,
        destination: &mut Vec<u32>,
    ) -> Result<(), ReadbackError> {
        destination.clear();
        if count == 0 {
            return Ok(());
        }
        let size = (count * std::mem::size_of::<u32>()) as u64;
        let staging = match self.staging.as_ref() {
            Some(buffer) if self.capacity >= size => buffer,
            _ => {
                // Grow by high-water mark rather than to the exact request, so
                // a layer that gains one slot does not force a reallocation.
                let capacity = size.max(self.capacity.saturating_mul(2)).max(4096);
                self.staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback staging (reused)"),
                    size: capacity,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.capacity = capacity;
                self.allocations += 1;
                self.staging.as_ref().ok_or(ReadbackError::Cancelled)?
            }
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback (reused)"),
        });
        encoder.copy_buffer_to_buffer(source, 0, staging, 0, size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(0..size);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            if sender.send(result).is_err() {
                log_dropped_readback();
            }
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(ReadbackError::Poll)?;
        receiver
            .recv()
            .map_err(|_| ReadbackError::Cancelled)?
            .map_err(ReadbackError::Map)?;

        {
            let view = slice.get_mapped_range().map_err(ReadbackError::Range)?;
            destination.reserve(count);
            for chunk in view.chunks_exact(4) {
                destination.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        staging.unmap();
        Ok(())
    }
}

/// The one place a dropped readback result is acknowledged.
///
/// `wgpui-wgpu` has no logging dependency yet and `AGENTS.md` forbids
/// discarding a fallible result silently, so this is the visible acknowledgement
/// until the crate gains one. It is reachable only if the device is destroyed
/// while a map is in flight.
fn log_dropped_readback() {
    eprintln!("wgpui-wgpu: buffer map completed after its receiver was dropped");
}
