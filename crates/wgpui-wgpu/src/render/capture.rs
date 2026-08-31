//! Capture-only GPU diagnostics for the native renderer.

use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The version of the native GPU capture records.
pub const CAPTURE_SCHEMA_VERSION: u32 = 1;

/// An identifier assigned only while a capture is collecting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceId(u64);

impl ResourceId {
    /// Returns the numeric identifier used by an inspector.
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// An opaque capture frame identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CaptureFrameId(u64);

impl CaptureFrameId {
    /// Returns the numeric identifier used by an inspector.
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// The capture request limits. Limits make capture overhead explicit and keep
/// a malformed or unusually large frame from creating an unbounded snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureRequest {
    /// Maximum number of timestamp scopes in one capture.
    pub timestamp_pairs: u32,
    /// Maximum command records retained by the capture.
    pub command_capacity: usize,
    /// Maximum resource records retained by the capture.
    pub resource_capacity: usize,
}

impl Default for CaptureRequest {
    fn default() -> Self {
        Self {
            timestamp_pairs: 256,
            command_capacity: 16_384,
            resource_capacity: 4_096,
        }
    }
}

/// GPU timestamp capability negotiated from the opened device.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimestampSupport {
    /// Whether the device exposes timestamp queries at all.
    pub timestamp_queries: bool,
    /// Whether arbitrary encoder timestamp commands are available.
    pub inside_encoders: bool,
    /// Whether pass timestamp commands are available.
    pub inside_passes: bool,
    /// Device timestamp ticks converted to nanoseconds.
    pub period_nanoseconds: f64,
}

impl TimestampSupport {
    /// Read support from the device and the queue's timestamp period.
    pub fn from_device(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let features = device.features();
        Self {
            timestamp_queries: features.contains(wgpu::Features::TIMESTAMP_QUERY),
            inside_encoders: features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
            inside_passes: features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES),
            period_nanoseconds: f64::from(queue.get_timestamp_period()),
        }
    }

    /// Support with no GPU timestamp capability.
    pub const fn unavailable() -> Self {
        Self {
            timestamp_queries: false,
            inside_encoders: false,
            inside_passes: false,
            period_nanoseconds: 0.0,
        }
    }

    fn can_write_encoder_timestamps(self) -> bool {
        self.timestamp_queries
            && self.inside_encoders
            && self.period_nanoseconds.is_finite()
            && self.period_nanoseconds > 0.0
    }
}

/// Why timestamp values are unavailable for a capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimestampUnavailableReason {
    /// The device does not expose timestamp queries.
    Unsupported,
    /// The device exposes queries but not the encoder write operation used by
    /// this renderer.
    EncoderWritesUnsupported,
    /// The bounded query pool was exhausted.
    QueryPoolExhausted,
    /// The device was lost before results became readable.
    DeviceLost,
    /// The driver rejected or cancelled a readback map.
    ReadbackFailed,
}

/// The result state of one GPU timestamp scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimestampResultState {
    /// A query pair was allocated and its result is not mapped yet.
    Pending,
    /// A delayed readback completed successfully.
    Ready,
    /// The scope could not be measured for the stated reason.
    Unavailable(TimestampUnavailableReason),
}

/// A named GPU scope and its explicit attribution.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuScopeRecord {
    /// Human-readable operation name.
    pub name: String,
    /// Query result index, when one was allocated.
    pub query_pair: Option<u32>,
    /// Raw start and end ticks after delayed readback.
    pub ticks: Option<(u64, u64)>,
    /// Converted duration in nanoseconds after readback.
    pub duration_nanoseconds: Option<f64>,
    /// CPU estimate in the calibrated capture timeline.
    pub cpu_start_nanoseconds: Option<u64>,
    /// Explicit association with a retained element or unknown work.
    pub attribution: Attribution,
    /// Why a timestamp is pending or unavailable.
    pub state: TimestampResultState,
}

/// Attribution for GPU work that may not have an element owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Attribution {
    /// The operation is associated with a retained element address.
    Element { address: u64, generation: u64 },
    /// The backend cannot prove an element association.
    Unknown,
}

/// A resource kind visible to a capture inspector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    /// A GPU buffer, including primitive and indirect argument buffers.
    Buffer,
    /// A texture or texture view backing a render target or atlas page.
    Texture,
    /// A sampler.
    Sampler,
    /// A bind-group layout.
    BindGroupLayout,
    /// A bind group.
    BindGroup,
    /// A render or compute pipeline.
    Pipeline,
    /// A timestamp query set or its resolve/readback buffer.
    Query,
}

/// Metadata for a resource referenced by recorded commands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceRecord {
    /// Capture-local identity.
    pub id: ResourceId,
    /// Resource category.
    pub kind: ResourceKind,
    /// Stable renderer-side label.
    pub label: String,
    /// Known byte size, when the resource is buffer-like.
    pub size: Option<u64>,
}

/// Load/store information for one color attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttachmentRecord {
    /// Capture-local attachment resource, if the caller registered one.
    pub resource: Option<ResourceId>,
    /// Whether the pass clears or preserves the attachment.
    pub load: LoadOperation,
    /// Whether the attachment is stored after the pass.
    pub store: bool,
}

/// Render-pass load operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadOperation {
    /// Clear before drawing.
    Clear,
    /// Preserve the previous contents.
    Load,
}

/// A RenderDoc-like command record. It describes the state transitions that
/// matter to inspection, rather than retaining a second executable command
/// stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommandRecord {
    /// Start a render pass.
    BeginRenderPass {
        label: String,
        attachments: Vec<AttachmentRecord>,
        viewport: [u32; 4],
        scissor: Option<[u32; 4]>,
    },
    /// End a render pass.
    EndRenderPass,
    /// Bind a pipeline.
    SetPipeline { pipeline: ResourceId },
    /// Bind a group and dynamic offsets.
    SetBindGroup {
        index: u32,
        bind_group: ResourceId,
        offsets: Vec<u32>,
    },
    /// Issue a direct draw.
    Draw {
        vertex_count: u32,
        instance_count: u32,
    },
    /// Issue an indirect draw.
    DrawIndirect { arguments: ResourceId, offset: u64 },
    /// Issue a multi-draw indirect operation.
    MultiDrawIndirect {
        arguments: ResourceId,
        offset: u64,
        count: u32,
        count_buffer: Option<ResourceId>,
    },
    /// Dispatch a compute operation.
    Dispatch { workgroups: [u32; 3] },
    /// Copy a buffer or texture resource.
    Copy {
        source: ResourceId,
        destination: ResourceId,
        size: u64,
    },
    /// Submit one command buffer to the queue.
    Submit { command_count: u32 },
    /// Present the frame.
    Present,
}

/// A calibrated relationship between CPU monotonic time and GPU ticks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClockCorrelation {
    /// CPU time sampled immediately before command encoding, in nanoseconds
    /// from the capture's monotonic origin.
    pub cpu_anchor_nanoseconds: u64,
    /// GPU tick used as the corresponding anchor.
    pub gpu_anchor_ticks: u64,
    /// GPU tick duration in nanoseconds.
    pub period_nanoseconds: f64,
}

impl ClockCorrelation {
    /// Convert a GPU tick into the approximate CPU capture timeline.
    pub fn gpu_to_cpu_nanoseconds(self, ticks: u64) -> Option<u64> {
        let delta = ticks as f64 - self.gpu_anchor_ticks as f64;
        let converted = self.cpu_anchor_nanoseconds as f64 + delta * self.period_nanoseconds;
        if converted.is_finite() && converted >= 0.0 && converted <= u64::MAX as f64 {
            Some(converted as u64)
        } else {
            None
        }
    }

    /// Convert a pair of GPU ticks to a non-negative duration.
    pub fn duration_nanoseconds(self, start: u64, end: u64) -> Option<f64> {
        if end < start || !self.period_nanoseconds.is_finite() || self.period_nanoseconds <= 0.0 {
            return None;
        }
        Some((end - start) as f64 * self.period_nanoseconds)
    }
}

/// An immutable capture after the presentation boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuCapture {
    /// Capture schema version.
    pub schema_version: u32,
    /// Captured frame identity.
    pub frame: CaptureFrameId,
    /// CPU capture interval.
    pub cpu_start_nanoseconds: u64,
    /// CPU submission timestamp.
    pub cpu_submit_nanoseconds: u64,
    /// CPU timestamp at the presentation boundary.
    pub cpu_end_nanoseconds: u64,
    /// Calibrated GPU/CPU relationship, when GPU timestamps are available.
    pub clock: Option<ClockCorrelation>,
    /// Resources created or referenced during collection.
    pub resources: Vec<ResourceRecord>,
    /// Command/state records in submission order.
    pub commands: Vec<CommandRecord>,
    /// GPU scopes, including pending and unavailable scopes.
    pub gpu_scopes: Vec<GpuScopeRecord>,
    /// Whether the device was lost during the capture.
    pub device_lost: Option<String>,
    /// Number of records dropped after the configured bounds.
    pub dropped_records: u64,
    /// Whether the capture has all currently submitted query results.
    pub timestamps_pending: bool,
}

#[derive(Debug)]
pub enum CaptureError {
    /// A capture is already armed or collecting.
    AlreadyActive,
    /// The request has no room for timestamp pairs.
    InvalidRequest,
    /// The device was lost while capture state was active.
    DeviceLost(String),
    /// A GPU readback failed.
    Readback(String),
}

impl std::fmt::Display for CaptureError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyActive => write!(formatter, "a GPU capture is already active"),
            Self::InvalidRequest => write!(formatter, "GPU capture request has invalid limits"),
            Self::DeviceLost(reason) => write!(formatter, "GPU capture device lost: {reason}"),
            Self::Readback(reason) => write!(formatter, "GPU capture readback failed: {reason}"),
        }
    }
}

impl std::error::Error for CaptureError {}

#[derive(Debug)]
struct TimestampPool {
    query_set: Option<wgpu::QuerySet>,
    capacity_pairs: u32,
    next_pair: u32,
}

impl TimestampPool {
    fn new(device: &wgpu::Device, capacity_pairs: u32, available: bool) -> Self {
        let query_set = if available {
            Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("wgpui capture timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: capacity_pairs.saturating_mul(2),
            }))
        } else {
            None
        };
        Self {
            query_set,
            capacity_pairs,
            next_pair: 0,
        }
    }

    fn allocate(&mut self) -> Option<(u32, u32, u32)> {
        if self.next_pair >= self.capacity_pairs {
            return None;
        }
        let pair = self.next_pair;
        self.next_pair += 1;
        Some((
            pair,
            pair.saturating_mul(2),
            pair.saturating_mul(2).saturating_add(1),
        ))
    }
}

#[derive(Debug)]
struct PendingReadback {
    buffer: wgpu::Buffer,
    receiver: Option<Receiver<Result<(), wgpu::BufferAsyncError>>>,
    scope_indices: Vec<(usize, u32, u32)>,
}

#[derive(Debug)]
struct ActiveCapture {
    frame: CaptureFrameId,
    cpu_start: Instant,
    cpu_anchor_nanoseconds: u64,
    request: CaptureRequest,
    resources: Vec<ResourceRecord>,
    commands: Vec<CommandRecord>,
    gpu_scopes: Vec<GpuScopeRecord>,
    pending_readbacks: VecDeque<PendingReadback>,
    timestamp_pool: TimestampPool,
    clock: Option<ClockCorrelation>,
    cpu_submit_nanoseconds: Option<u64>,
    dropped_records: u64,
    device_lost: Option<String>,
}

/// Capture state owned by a native frame renderer.
#[derive(Debug)]
pub struct GpuCaptureAdapter {
    support: TimestampSupport,
    next_frame: u64,
    active: Option<ActiveCapture>,
    frozen: Option<GpuCapture>,
    pending_readbacks: VecDeque<PendingReadback>,
    device_lost: Option<String>,
    device_loss_signal: Arc<Mutex<Option<String>>>,
    device_loss_handler_installed: bool,
}

impl GpuCaptureAdapter {
    /// Create an inert adapter. It allocates no GPU resource.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            support: TimestampSupport::from_device(device, queue),
            next_frame: 0,
            active: None,
            frozen: None,
            pending_readbacks: VecDeque::new(),
            device_lost: None,
            device_loss_signal: Arc::new(Mutex::new(None)),
            device_loss_handler_installed: false,
        }
    }

    /// Create an adapter for tests or a caller that already negotiated support.
    pub fn with_support(support: TimestampSupport) -> Self {
        Self {
            support,
            next_frame: 0,
            active: None,
            frozen: None,
            pending_readbacks: VecDeque::new(),
            device_lost: None,
            device_loss_signal: Arc::new(Mutex::new(None)),
            device_loss_handler_installed: false,
        }
    }

    /// Whether this device can record encoder timestamp commands.
    pub const fn timestamp_support(&self) -> TimestampSupport {
        self.support
    }

    /// Return a device-loss description observed by the capture.
    pub fn device_lost_reason(&self) -> Option<&str> {
        self.device_lost.as_deref()
    }

    /// Arm the next frame. Query and readback resources are allocated here, not
    /// at adapter construction or in the normal render loop.
    pub fn arm(
        &mut self,
        device: &wgpu::Device,
        request: CaptureRequest,
    ) -> Result<(), CaptureError> {
        if self.active.is_some() {
            return Err(CaptureError::AlreadyActive);
        }
        if request.timestamp_pairs == 0
            || request.timestamp_pairs > wgpu::QUERY_SET_MAX_QUERIES / 2
            || request.command_capacity == 0
            || request.resource_capacity == 0
        {
            return Err(CaptureError::InvalidRequest);
        }
        self.frozen = None;
        self.device_lost = None;
        self.install_device_loss_handler(device);
        let frame = CaptureFrameId(self.next_frame);
        self.next_frame = self.next_frame.saturating_add(1);
        let cpu_start = Instant::now();
        let timestamp_pool = TimestampPool::new(
            device,
            request.timestamp_pairs,
            self.support.can_write_encoder_timestamps(),
        );
        let has_query_pool = timestamp_pool.query_set.is_some();
        let mut active = ActiveCapture {
            frame,
            cpu_start,
            cpu_anchor_nanoseconds: 0,
            request,
            resources: Vec::new(),
            commands: Vec::new(),
            gpu_scopes: Vec::new(),
            pending_readbacks: VecDeque::new(),
            timestamp_pool,
            clock: None,
            cpu_submit_nanoseconds: None,
            dropped_records: 0,
            device_lost: None,
        };
        if has_query_pool {
            active.resources.push(ResourceRecord {
                id: ResourceId(0),
                kind: ResourceKind::Query,
                label: "capture timestamp query pool".to_string(),
                size: Some(u64::from(request.timestamp_pairs).saturating_mul(16)),
            });
        }
        self.active = Some(active);
        Ok(())
    }

    fn install_device_loss_handler(&mut self, device: &wgpu::Device) {
        if self.device_loss_handler_installed {
            return;
        }
        let signal = Arc::clone(&self.device_loss_signal);
        device.set_device_lost_callback(move |_reason, description| {
            if let Ok(mut pending_reason) = signal.lock() {
                *pending_reason = Some(description);
            }
        });
        self.device_loss_handler_installed = true;
    }

    fn take_device_loss_signal(&self) -> Option<String> {
        let Ok(mut signal) = self.device_loss_signal.lock() else {
            return Some("device-loss callback state was poisoned".to_string());
        };
        signal.take()
    }

    /// Start collecting the current frame. This transitions an armed capture
    /// into its one-frame collection state.
    pub fn begin_frame(&mut self) -> Option<CaptureFrameId> {
        let active = self.active.as_mut()?;
        active.cpu_anchor_nanoseconds = active
            .cpu_start
            .elapsed()
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64;
        Some(active.frame)
    }

    /// Mark the device lost. The capture remains inspectable and timestamp
    /// results become explicitly unavailable rather than being fabricated.
    pub fn mark_device_lost(&mut self, reason: impl Into<String>) {
        let reason = reason.into();
        self.device_lost = Some(reason.clone());
        if let Some(active) = self.active.as_mut() {
            active.device_lost = Some(reason.clone());
            for scope in &mut active.gpu_scopes {
                if scope.ticks.is_none() {
                    scope.state =
                        TimestampResultState::Unavailable(TimestampUnavailableReason::DeviceLost);
                }
            }
        }
        if let Some(capture) = self.frozen.as_mut() {
            capture.device_lost = Some(reason.clone());
            for scope in &mut capture.gpu_scopes {
                if scope.ticks.is_none() {
                    scope.state =
                        TimestampResultState::Unavailable(TimestampUnavailableReason::DeviceLost);
                }
            }
        }
    }

    /// Register a resource in the capture and return its local identity.
    pub fn register_resource(
        &mut self,
        kind: ResourceKind,
        label: impl Into<String>,
        size: Option<u64>,
    ) -> Option<ResourceId> {
        let active = self.active.as_mut()?;
        if active.resources.len() >= active.request.resource_capacity {
            active.dropped_records = active.dropped_records.saturating_add(1);
            return None;
        }
        let id = ResourceId(active.resources.len() as u64);
        active.resources.push(ResourceRecord {
            id,
            kind,
            label: label.into(),
            size,
        });
        Some(id)
    }

    fn command(&mut self, command: CommandRecord) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        if active.commands.len() >= active.request.command_capacity {
            active.dropped_records = active.dropped_records.saturating_add(1);
            return;
        }
        active.commands.push(command);
    }

    /// Record a render pass boundary.
    pub fn record_begin_render_pass(
        &mut self,
        label: impl Into<String>,
        attachments: Vec<AttachmentRecord>,
        viewport: [u32; 4],
        scissor: Option<[u32; 4]>,
    ) {
        self.command(CommandRecord::BeginRenderPass {
            label: label.into(),
            attachments,
            viewport,
            scissor,
        });
    }

    /// Record a render pass end.
    pub fn record_end_render_pass(&mut self) {
        self.command(CommandRecord::EndRenderPass);
    }

    /// Record a pipeline bind.
    pub fn record_set_pipeline(&mut self, pipeline: ResourceId) {
        self.command(CommandRecord::SetPipeline { pipeline });
    }

    /// Record a bind-group bind.
    pub fn record_set_bind_group(&mut self, index: u32, bind_group: ResourceId, offsets: &[u32]) {
        self.command(CommandRecord::SetBindGroup {
            index,
            bind_group,
            offsets: offsets.to_vec(),
        });
    }

    /// Record a direct draw.
    pub fn record_draw(&mut self, vertex_count: u32, instance_count: u32) {
        self.command(CommandRecord::Draw {
            vertex_count,
            instance_count,
        });
    }

    /// Record an indirect draw.
    pub fn record_draw_indirect(&mut self, arguments: ResourceId, offset: u64) {
        self.command(CommandRecord::DrawIndirect { arguments, offset });
    }

    /// Record a multi-draw indirect operation and optional GPU count buffer.
    pub fn record_multi_draw_indirect(
        &mut self,
        arguments: ResourceId,
        offset: u64,
        count: u32,
        count_buffer: Option<ResourceId>,
    ) {
        self.command(CommandRecord::MultiDrawIndirect {
            arguments,
            offset,
            count,
            count_buffer,
        });
    }

    /// Record a compute dispatch.
    pub fn record_dispatch(&mut self, workgroups: [u32; 3]) {
        self.command(CommandRecord::Dispatch { workgroups });
    }

    /// Record a resource copy.
    pub fn record_copy(&mut self, source: ResourceId, destination: ResourceId, size: u64) {
        self.command(CommandRecord::Copy {
            source,
            destination,
            size,
        });
    }

    /// Allocate and write a timestamp pair around encoder commands.
    pub fn write_timestamp_scope(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        name: impl Into<String>,
        attribution: Attribution,
    ) -> Option<usize> {
        let active = self.active.as_mut()?;
        let name = name.into();
        if !self.support.timestamp_queries {
            active.gpu_scopes.push(GpuScopeRecord {
                name,
                query_pair: None,
                ticks: None,
                duration_nanoseconds: None,
                cpu_start_nanoseconds: None,
                attribution,
                state: TimestampResultState::Unavailable(TimestampUnavailableReason::Unsupported),
            });
            return None;
        }
        if !self.support.inside_encoders {
            active.gpu_scopes.push(GpuScopeRecord {
                name,
                query_pair: None,
                ticks: None,
                duration_nanoseconds: None,
                cpu_start_nanoseconds: None,
                attribution,
                state: TimestampResultState::Unavailable(
                    TimestampUnavailableReason::EncoderWritesUnsupported,
                ),
            });
            return None;
        }
        let Some((pair, start, _end)) = active.timestamp_pool.allocate() else {
            active.gpu_scopes.push(GpuScopeRecord {
                name,
                query_pair: None,
                ticks: None,
                duration_nanoseconds: None,
                cpu_start_nanoseconds: None,
                attribution,
                state: TimestampResultState::Unavailable(
                    TimestampUnavailableReason::QueryPoolExhausted,
                ),
            });
            return None;
        };
        let query_set = active.timestamp_pool.query_set.as_ref()?;
        encoder.write_timestamp(query_set, start);
        let index = active.gpu_scopes.len();
        active.gpu_scopes.push(GpuScopeRecord {
            name,
            query_pair: Some(pair),
            ticks: None,
            duration_nanoseconds: None,
            cpu_start_nanoseconds: None,
            attribution,
            state: TimestampResultState::Pending,
        });
        Some(index)
    }

    /// Finish a previously allocated timestamp scope.
    pub fn end_timestamp_scope(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        scope: Option<usize>,
    ) {
        let Some(scope_index) = scope else {
            return;
        };
        let Some(active) = self.active.as_mut() else {
            return;
        };
        let Some(scope_record) = active.gpu_scopes.get(scope_index) else {
            return;
        };
        let Some(pair) = scope_record.query_pair else {
            return;
        };
        let Some(query_set) = active.timestamp_pool.query_set.as_ref() else {
            return;
        };
        encoder.write_timestamp(query_set, pair.saturating_mul(2).saturating_add(1));
    }

    /// Resolve the current query pool into a delayed, asynchronously mapped
    /// readback buffer. The buffer is retained until `poll` observes completion.
    pub fn resolve_readback(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        let Some(query_set) = active.timestamp_pool.query_set.as_ref() else {
            return;
        };
        let used_queries = active.timestamp_pool.next_pair.saturating_mul(2);
        if used_queries == 0 {
            return;
        }
        let size = u64::from(used_queries).saturating_mul(8);
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpui capture timestamp resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpui capture timestamp readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.resolve_query_set(query_set, 0..used_queries, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, size);
        let scope_indices = active
            .gpu_scopes
            .iter()
            .enumerate()
            .filter_map(|(index, scope)| {
                scope.query_pair.map(|pair| {
                    (
                        index,
                        pair.saturating_mul(2),
                        pair.saturating_mul(2).saturating_add(1),
                    )
                })
            })
            .collect();
        active.pending_readbacks.push_back(PendingReadback {
            buffer: readback,
            receiver: None,
            scope_indices,
        });
    }

    /// Begin asynchronous mapping after the command buffer containing the
    /// resolve and copy has been submitted.
    pub fn start_readback_maps(&mut self) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        for pending in &mut active.pending_readbacks {
            if pending.receiver.is_some() {
                continue;
            }
            let slice = pending.buffer.slice(..);
            let (sender, receiver) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                if sender.send(result).is_err() {
                    log::warn!("wgpui-wgpu: capture timestamp readback completed after its receiver was dropped");
                }
            });
            pending.receiver = Some(receiver);
        }
    }

    /// Record CPU submission and queue submission metadata.
    pub fn record_submit(&mut self, command_count: u32) {
        if let Some(active) = self.active.as_mut() {
            active.cpu_submit_nanoseconds = Some(
                active
                    .cpu_start
                    .elapsed()
                    .as_nanos()
                    .min(u128::from(u64::MAX)) as u64,
            );
        }
        self.command(CommandRecord::Submit { command_count });
    }

    /// Record the presentation boundary and freeze the active frame.
    pub fn finish_frame(&mut self) {
        self.command(CommandRecord::Present);
        let Some(active) = self.active.take() else {
            return;
        };
        let cpu_end = active
            .cpu_start
            .elapsed()
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64;
        let pending = !active.pending_readbacks.is_empty();
        self.pending_readbacks.extend(active.pending_readbacks);
        let capture = GpuCapture {
            schema_version: CAPTURE_SCHEMA_VERSION,
            frame: active.frame,
            cpu_start_nanoseconds: 0,
            cpu_submit_nanoseconds: active.cpu_submit_nanoseconds.unwrap_or(cpu_end),
            cpu_end_nanoseconds: cpu_end,
            clock: active.clock,
            resources: active.resources,
            commands: active.commands,
            gpu_scopes: active.gpu_scopes,
            device_lost: active.device_lost.or_else(|| self.device_lost.clone()),
            dropped_records: active.dropped_records,
            timestamps_pending: pending,
        };
        self.frozen = Some(capture);
    }

    /// Poll delayed timestamp readbacks without blocking the frame loop.
    pub fn poll(&mut self, device: &wgpu::Device) -> Result<(), CaptureError> {
        if let Some(reason) = self.take_device_loss_signal() {
            self.mark_device_lost(reason.clone());
            self.pending_readbacks.clear();
            if let Some(capture) = self.frozen.as_mut() {
                capture.timestamps_pending = false;
            }
            return Err(CaptureError::DeviceLost(reason));
        }
        if let Some(reason) = self.device_lost.clone() {
            self.pending_readbacks.clear();
            if let Some(capture) = self.frozen.as_mut() {
                capture.timestamps_pending = false;
            }
            return Err(CaptureError::DeviceLost(reason));
        }
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| CaptureError::Readback(error.to_string()))?;
        while let Some(pending) = self.pending_readbacks.pop_front() {
            let Some(receiver) = pending.receiver.as_ref() else {
                self.pending_readbacks.push_front(pending);
                break;
            };
            match receiver.try_recv() {
                Ok(Ok(())) => {
                    let view = pending
                        .buffer
                        .slice(..)
                        .get_mapped_range()
                        .map_err(|error| CaptureError::Readback(error.to_string()))?;
                    for (scope_index, start_index, end_index) in pending.scope_indices {
                        let start_offset = usize::try_from(start_index)
                            .unwrap_or(usize::MAX)
                            .saturating_mul(8);
                        let end_offset = usize::try_from(end_index)
                            .unwrap_or(usize::MAX)
                            .saturating_mul(8);
                        let Some(start_bytes) =
                            view.get(start_offset..start_offset.saturating_add(8))
                        else {
                            continue;
                        };
                        let Some(end_bytes) = view.get(end_offset..end_offset.saturating_add(8))
                        else {
                            continue;
                        };
                        let start = u64::from_le_bytes(start_bytes.try_into().map_err(|_| {
                            CaptureError::Readback("invalid timestamp result".to_string())
                        })?);
                        let end = u64::from_le_bytes(end_bytes.try_into().map_err(|_| {
                            CaptureError::Readback("invalid timestamp result".to_string())
                        })?);
                        let Some(capture) = self.frozen.as_mut() else {
                            continue;
                        };
                        if let Some(scope) = capture.gpu_scopes.get_mut(scope_index) {
                            scope.ticks = Some((start, end));
                            scope.duration_nanoseconds = ClockCorrelation {
                                cpu_anchor_nanoseconds: capture.cpu_start_nanoseconds,
                                gpu_anchor_ticks: start,
                                period_nanoseconds: self.support.period_nanoseconds,
                            }
                            .duration_nanoseconds(start, end);
                            scope.cpu_start_nanoseconds = ClockCorrelation {
                                cpu_anchor_nanoseconds: capture.cpu_start_nanoseconds,
                                gpu_anchor_ticks: start,
                                period_nanoseconds: self.support.period_nanoseconds,
                            }
                            .gpu_to_cpu_nanoseconds(start);
                            scope.state = TimestampResultState::Ready;
                            capture.clock = Some(ClockCorrelation {
                                cpu_anchor_nanoseconds: capture.cpu_start_nanoseconds,
                                gpu_anchor_ticks: start,
                                period_nanoseconds: self.support.period_nanoseconds,
                            });
                        }
                    }
                    drop(view);
                    pending.buffer.unmap();
                }
                Ok(Err(_)) => {
                    for (scope_index, _, _) in pending.scope_indices {
                        if let Some(capture) = self.frozen.as_mut()
                            && let Some(scope) = capture.gpu_scopes.get_mut(scope_index)
                        {
                            scope.state = TimestampResultState::Unavailable(
                                TimestampUnavailableReason::ReadbackFailed,
                            );
                        }
                    }
                }
                Err(TryRecvError::Empty) => {
                    self.pending_readbacks.push_front(pending);
                    break;
                }
                Err(TryRecvError::Disconnected) => {
                    for (scope_index, _, _) in pending.scope_indices {
                        if let Some(capture) = self.frozen.as_mut()
                            && let Some(scope) = capture.gpu_scopes.get_mut(scope_index)
                        {
                            scope.state = TimestampResultState::Unavailable(
                                TimestampUnavailableReason::DeviceLost,
                            );
                        }
                    }
                }
            }
        }
        if self.pending_readbacks.is_empty()
            && let Some(capture) = self.frozen.as_mut()
        {
            capture.timestamps_pending = false;
        }
        Ok(())
    }

    /// Poll a frozen capture's delayed results and return a clone when ready.
    pub fn take_capture(&mut self) -> Option<GpuCapture> {
        self.frozen.take()
    }

    /// Return the frozen capture without consuming it.
    pub fn capture(&self) -> Option<&GpuCapture> {
        self.frozen.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_timestamp_support_is_explicit() {
        let adapter = GpuCaptureAdapter::with_support(TimestampSupport::unavailable());
        assert!(!adapter.timestamp_support().timestamp_queries);
        assert!(!adapter.timestamp_support().can_write_encoder_timestamps());
    }

    #[test]
    fn clock_correlation_converts_ticks_and_durations() {
        let clock = ClockCorrelation {
            cpu_anchor_nanoseconds: 1_000,
            gpu_anchor_ticks: 20,
            period_nanoseconds: 2.5,
        };
        assert_eq!(clock.gpu_to_cpu_nanoseconds(24), Some(1_010));
        assert_eq!(clock.duration_nanoseconds(20, 24), Some(10.0));
        assert_eq!(clock.duration_nanoseconds(24, 20), None);
    }

    #[test]
    fn attribution_does_not_infer_an_element_for_backend_work() {
        assert_eq!(Attribution::Unknown, Attribution::Unknown);
        assert_ne!(
            Attribution::Unknown,
            Attribution::Element {
                address: 1,
                generation: 1,
            }
        );
    }

    #[test]
    fn timestamp_pool_reports_exhaustion_without_overallocating() {
        let mut pool = TimestampPool {
            query_set: None,
            capacity_pairs: 1,
            next_pair: 0,
        };
        assert_eq!(pool.allocate(), Some((0, 0, 1)));
        assert_eq!(pool.allocate(), None);
        assert_eq!(pool.next_pair, 1);
    }

    #[test]
    fn device_loss_is_retained_as_capture_metadata() {
        let mut adapter = GpuCaptureAdapter::with_support(TimestampSupport::unavailable());
        adapter.mark_device_lost("test loss");
        assert_eq!(adapter.device_lost_reason(), Some("test loss"));
    }

    #[test]
    fn native_capture_records_submit_and_delivers_delayed_timestamps() {
        let Some(context) = crate::render::device::context_or_report("GPU capture") else {
            return;
        };
        let mut adapter = GpuCaptureAdapter::new(&context.device, &context.queue);
        adapter
            .arm(
                &context.device,
                CaptureRequest {
                    timestamp_pairs: 1,
                    command_capacity: 32,
                    resource_capacity: 32,
                },
            )
            .expect("capture should arm");
        assert!(adapter.begin_frame().is_some());
        let pipeline = adapter
            .register_resource(ResourceKind::Pipeline, "capture test pipeline", None)
            .expect("pipeline resource should be recorded");
        let bind_group = adapter
            .register_resource(ResourceKind::BindGroup, "capture test bind group", None)
            .expect("bind group resource should be recorded");
        let arguments = adapter
            .register_resource(
                ResourceKind::Buffer,
                "capture test indirect arguments",
                Some(16),
            )
            .expect("argument resource should be recorded");
        adapter.record_begin_render_pass(
            "capture test pass",
            vec![AttachmentRecord {
                resource: None,
                load: LoadOperation::Clear,
                store: true,
            }],
            [0, 0, 1, 1],
            None,
        );
        adapter.record_set_pipeline(pipeline);
        adapter.record_set_bind_group(0, bind_group, &[]);
        adapter.record_multi_draw_indirect(arguments, 0, 1, None);
        adapter.record_end_render_pass();
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("capture test"),
            });
        let scope =
            adapter.write_timestamp_scope(&mut encoder, "capture test scope", Attribution::Unknown);
        adapter.end_timestamp_scope(&mut encoder, scope);
        adapter.resolve_readback(&context.device, &mut encoder);
        adapter.record_submit(1);
        context.queue.submit(Some(encoder.finish()));
        adapter.start_readback_maps();
        adapter.finish_frame();

        let Some(capture) = adapter.capture() else {
            return;
        };
        assert!(
            capture
                .commands
                .iter()
                .any(|command| matches!(command, CommandRecord::Submit { .. }))
        );
        assert!(
            capture
                .commands
                .iter()
                .any(|command| matches!(command, CommandRecord::BeginRenderPass { .. }))
        );
        assert!(
            capture
                .commands
                .iter()
                .any(|command| matches!(command, CommandRecord::SetPipeline { .. }))
        );
        assert!(
            capture
                .commands
                .iter()
                .any(|command| matches!(command, CommandRecord::SetBindGroup { .. }))
        );
        assert!(
            capture
                .commands
                .iter()
                .any(|command| matches!(command, CommandRecord::MultiDrawIndirect { .. }))
        );
        if adapter.timestamp_support().timestamp_queries
            && adapter.timestamp_support().inside_encoders
        {
            assert!(
                capture
                    .resources
                    .iter()
                    .any(|resource| resource.kind == ResourceKind::Query)
            );
        }
        assert_eq!(capture.gpu_scopes.len(), 1);
        let Some(scope) = capture.gpu_scopes.first() else {
            return;
        };
        assert_eq!(scope.attribution, Attribution::Unknown);
        if adapter.timestamp_support().timestamp_queries
            && adapter.timestamp_support().inside_encoders
        {
            assert!(capture.timestamps_pending);
            for _ in 0..8 {
                context
                    .device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .expect("GPU capture test polling should succeed");
                adapter
                    .poll(&context.device)
                    .expect("capture polling should succeed");
                if !adapter
                    .capture()
                    .is_some_and(|capture| capture.timestamps_pending)
                {
                    break;
                }
            }
            let capture = adapter.capture().expect("frozen capture remains available");
            assert!(!capture.timestamps_pending);
            let Some(scope) = capture.gpu_scopes.first() else {
                return;
            };
            assert!(scope.ticks.is_some());
            assert_eq!(scope.state, TimestampResultState::Ready);
        } else {
            let Some(scope) = capture.gpu_scopes.first() else {
                return;
            };
            assert_eq!(
                scope.state,
                TimestampResultState::Unavailable(
                    if adapter.timestamp_support().timestamp_queries {
                        TimestampUnavailableReason::EncoderWritesUnsupported
                    } else {
                        TimestampUnavailableReason::Unsupported
                    },
                )
            );
        }
    }
}
