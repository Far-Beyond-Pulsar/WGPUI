//! GPU residency for packed layers (retained rendering Pillar III, epic #84,
//! spec #94): the renderer-side registry that turns [`crate::scene_pack`]'s
//! per-layer packing into resident instance data, plus the upload-decision
//! bookkeeping that keeps an idle window at zero `write_buffer` calls.
//!
//! # Ownership
//!
//! The registry lives inside `WgpuRenderer`, keyed by [`LayerKey`]. Nothing
//! about it crosses a Platform trait: the window side only stamps content
//! tokens and emits scene spans; this module owns the allocator, the
//! transform-slot table, and every decision about when bytes move.
//!
//! # The decision boundary is pure
//!
//! [`SlabRegistry::plan_sync`] answers "does drawing this layer's marker
//! require touching any buffer?" from plain data — no device, no queue — so
//! tests can assert exact upload behaviour without a GPU. The renderer
//! executes a plan as one `write_buffer` per kind, byte offset =
//! `SlabRange.base * stride` (element-unit math owned by `slab.rs`; a
//! misaligned upload is unexpressible).
//!
//! # Fail-loud policy
//!
//! Every path that cannot draw a span correctly — allocator overflow, an
//! atlas page evicted under a resident sprite run — poisons or skips that
//! layer for the frame, warns once, bumps a counter, and posts a re-record
//! request so the next frame rebuilds the layer through the legacy paint
//! path. Wrong pixels are never shipped silently; a missing panel for one
//! frame is the documented last resort.

use std::sync::LazyLock;

use collections::{FxHashMap, FxHashSet};
use parking_lot::Mutex;

use crate::platform::cross::slab::{
    self, CompactionPlan, LayerSlabs, SlabAllocator, SlabKind, SlabOverflow,
};
use crate::{AtlasTextureId, LayerKey};

/// Whether advisory compaction may run during idle frames.
///
/// Correctness never depends on it; `WGPUI_SLAB_COMPACTION=0` turns it off.
pub(crate) fn compaction_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var("WGPUI_SLAB_COMPACTION")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(true)
    });
    *ENABLED
}

// ---------------------------------------------------------------------
// Counters.
// ---------------------------------------------------------------------

pub(crate) const COUNTER_BYTES_UPLOADED: &str = "slab: bytes uploaded";
pub(crate) const COUNTER_LAYERS_REALLOCATED: &str = "slab: layers reallocated";
pub(crate) const COUNTER_COMPACTIONS: &str = "slab: compactions";
pub(crate) const COUNTER_DRAW_CALLS: &str = "slab: draw calls";
pub(crate) const COUNTER_COMPACTION_PLANS_DEFERRED: &str = "slab: compaction plans deferred";
pub(crate) const COUNTER_ZERO_MOVE_PLANS: &str = "slab: compaction plans moved nothing";
const COUNTER_SPANS_DRAWN_CLEAN: &str = "slab: spans drawn clean";
const COUNTER_SPANS_SKIPPED_EVICTED: &str = "slab: spans skipped (awaiting re-record)";
const COUNTER_SYNC_OVERFLOWS: &str = "slab: sync overflowed";
const COUNTER_TRANSFORM_SLOTS_WRITTEN: &str = "slab: transform slots written";
const COUNTER_REGISTRY_GC_FREED: &str = "slab: registry entries gc'd";
const COUNTER_EVICTION_POISONED: &str = "slab: layers poisoned by atlas eviction";

// ---------------------------------------------------------------------
// Cross-component re-record requests.
//
// Atlas eviction happens deep inside texture cache code holding no handle to
// the owning window; the renderer discovers it at frame start. Neither can
// reach `WindowInvalidator` directly, so poison flows through this
// process-global queue that `Window::draw` drains before recording the next
// frame, which is what makes invalidation-before-draw ordering hold: the
// affected layer rebuilds (fresh tiles, fresh token) before its slab is
// drawn again.
// ---------------------------------------------------------------------

static LAYERS_NEEDING_RERECORD: LazyLock<Mutex<Vec<LayerKey>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

/// Post a request to rebuild `keys` through the legacy paint path next frame.
pub(crate) fn request_rerecord(keys: impl IntoIterator<Item = LayerKey>) {
    let mut queue = LAYERS_NEEDING_RERECORD.lock();
    queue.extend(keys);
}

/// Drain every pending re-record request. Called once per window draw.
pub(crate) fn take_rerecord_requests() -> Vec<LayerKey> {
    std::mem::take(&mut LAYERS_NEEDING_RERECORD.lock())
}

// ---------------------------------------------------------------------
// Transform uniforms.
// ---------------------------------------------------------------------

/// One layer's translate, as a 64-byte uniform slot. The padding fields make
/// the slot size fixed regardless of what alignment a driver reports for
/// dynamic offsets; the buffer strides slots at
/// `max(min_uniform_buffer_offset_alignment, 64)` anyway.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuLayerTransform {
    pub translate: [f32; 2],
    pub _pad0: [f32; 2],
    pub _pad1: [f32; 4],
    pub _pad2: [f32; 4],
    pub _pad3: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuLayerTransform>() == 64);

impl Default for GpuLayerTransform {
    fn default() -> Self {
        GpuLayerTransform {
            translate: [0.0, 0.0],
            _pad0: [0.0; 2],
            _pad1: [0.0; 4],
            _pad2: [0.0; 4],
            _pad3: [0.0; 4],
        }
    }
}

/// Assigns 64-byte transform slots to layers and tracks which are dirty.
///
/// Slot 0 stays permanently zero: legacy draws bind the transform bind group
/// with dynamic offset 0 and must see identity. Packed content is stored
/// relative to its layer's origin and restored in-shader by adding the
/// layer's slot, so a moved layer costs exactly one 64-byte write.
#[derive(Default)]
pub(crate) struct TransformTable {
    slot_of: FxHashMap<LayerKey, u32>,
    free_slots: Vec<u32>,
    values: Vec<GpuLayerTransform>,
    dirty: FxHashSet<u32>,
    /// Bounded scratch for [`TransformTable::drain_dirty_into`], so steady
    /// state drains never allocate.
    drain_scratch: Vec<u32>,
}

impl TransformTable {
    pub fn new() -> Self {
        let mut table = TransformTable::default();
        // Slot 0: the identity slot legacy draws read. Never assigned, never dirty.
        table.values.push(GpuLayerTransform::default());
        table
    }

    pub fn slot_count(&self) -> u32 {
        self.values.len() as u32
    }

    /// The slot for `key`, assigning one if needed.
    pub fn slot_for(&mut self, key: LayerKey) -> u32 {
        if let Some(&slot) = self.slot_of.get(&key) {
            return slot;
        }
        let slot = match self.free_slots.pop() {
            Some(slot) => slot,
            None => {
                let slot = self.values.len() as u32;
                self.values.push(GpuLayerTransform::default());
                slot
            }
        };
        self.slot_of.insert(key, slot);
        slot
    }

    /// Set `key`'s translate, reporting whether the GPU copy is stale.
    pub fn set_translate(&mut self, key: LayerKey, translate: [f32; 2]) -> bool {
        let slot = self.slot_for(key);
        let value = &mut self.values[slot as usize];
        if value.translate == translate {
            return false;
        }
        value.translate = translate;
        self.dirty.insert(slot);
        true
    }

    #[cfg(test)]
    pub fn slot_value(&self, slot: u32) -> GpuLayerTransform {
        self.values[slot as usize]
    }

    /// Take the dirty set, paired with each slot's current value.
    #[cfg(test)]
    pub fn drain_dirty(&mut self) -> Vec<(u32, GpuLayerTransform)> {
        let mut out = Vec::new();
        self.drain_dirty_into(&mut out);
        out
    }

    /// [`Self::drain_dirty`] into caller-owned storage: after a warm-up drain,
    /// the vector's capacity is reused every frame instead of reallocating.
    pub fn drain_dirty_into(&mut self, out: &mut Vec<(u32, GpuLayerTransform)>) {
        out.clear();
        let mut dirty = std::mem::take(&mut self.drain_scratch);
        dirty.clear();
        dirty.extend(self.dirty.drain());
        dirty.sort_unstable();
        out.extend(dirty.iter().map(|&slot| (slot, self.values[slot as usize])));
        self.drain_scratch = dirty;
    }

    /// Mark every occupied slot for re-upload (uniform buffer recreated).
    pub fn mark_all_dirty(&mut self) {
        // Slot 0 holds the permanent identity legacy draws read; it is already
        // zero, so re-marking it would only cost a redundant uniform write.
        self.dirty.extend(1..self.values.len() as u32);
    }

    pub fn release(&mut self, key: LayerKey) {
        if let Some(slot) = self.slot_of.remove(&key) {
            self.values[slot as usize] = GpuLayerTransform::default();
            self.free_slots.push(slot);
        }
    }
}

// ---------------------------------------------------------------------
// The registry proper.
// ---------------------------------------------------------------------

/// How long a registry entry may go unreferenced by any scene span before its
/// GPU state is reclaimed. Long enough that ordinary occluded-layer gaps
/// never churn; a freed-then-referenced layer simply re-uploads from the
/// marker's own bytes, so correctness never depends on the interval.
const GC_IDLE_FRAMES: u64 = 600;

/// Compaction cooldown schedule. A plan that moves nothing means the arenas
/// are sparse but already packed — exactly the steady state of an idle
/// window — so planning again immediately would rebuild the same empty plan
/// every frame. The first zero-move plan imposes a one-frame cooldown that
/// doubles per consecutive zero-move plan, capping at
/// `COMPACTION_BACKOFF_BASE_FRAMES << COMPACTION_BACKOFF_MAX_SHIFT` frames.
/// Any plan that moves something resets the schedule.
const COMPACTION_BACKOFF_BASE_FRAMES: u32 = 1;
const COMPACTION_BACKOFF_MAX_SHIFT: u32 = 8;

#[cfg(test)]
pub(crate) fn compaction_backoff_cap_frames() -> u32 {
    COMPACTION_BACKOFF_BASE_FRAMES << COMPACTION_BACKOFF_MAX_SHIFT
}

struct RegistryEntry {
    content_token: u64,
    slabs: LayerSlabs,
    counts: [u32; SlabKind::COUNT],
    uploaded_generation: Option<u64>,
    poisoned: bool,
    awaiting_rerecord: bool,
    referenced_pages: FxHashSet<(u32, crate::AtlasTextureKind)>,
    last_referenced_frame: u64,
    transform_slot: u32,
}

/// What the renderer must do to make a layer's span drawable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SyncPlan {
    /// Resident bytes already match the marker; touch nothing.
    Clean,
    /// Re-upload every occupied kind of this layer from the marker's bytes.
    UploadAllOccupied,
}

/// Renderer-owned slab state: the allocator plus per-layer residency records.
pub(crate) struct SlabRegistry {
    allocator: SlabAllocator<LayerKey>,
    entries: FxHashMap<LayerKey, RegistryEntry>,
    transforms: TransformTable,
    frame: u64,
    /// Remaining frames during which advisory compaction is suppressed by the
    /// zero-move backoff. Correctness never depends on compaction running, so
    /// a nonzero value only delays defragmentation.
    compaction_cooldown_frames: u32,
    /// Consecutive plans that moved nothing; drives the exponential backoff.
    zero_move_streak: u32,
    /// Layers uploaded since the last time the compaction gate was evaluated.
    /// Compaction planning is skipped while upload traffic is flowing: fresh
    /// allocations make any plan instantly stale and the CPU work competes
    /// with the upload path.
    uploads_since_last_plan: u32,
}

impl Default for SlabRegistry {
    fn default() -> Self {
        SlabRegistry {
            allocator: SlabAllocator::new(),
            entries: FxHashMap::default(),
            transforms: TransformTable::new(),
            frame: 0,
            compaction_cooldown_frames: 0,
            zero_move_streak: 0,
            uploads_since_last_plan: 0,
        }
    }
}

impl SlabRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance the frame counter and reclaim entries no span has referenced.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
        let stale: Vec<LayerKey> = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                self.frame.saturating_sub(entry.last_referenced_frame) > GC_IDLE_FRAMES
            })
            .map(|(key, _)| *key)
            .collect();
        let freed = stale.len();
        for key in stale {
            self.drop_entry(&key);
        }
        if freed > 0 {
            crate::render_stats::add(COUNTER_REGISTRY_GC_FREED, freed as u64);
        }
    }

    /// Decide whether drawing a span for (`key`, `content_token`) needs work.
    ///
    /// On anything other than [`SyncPlan::Clean`] the layer's slab ranges are
    /// brought in line with `counts` first (allocating, resizing, relocating),
    /// so a returned plan is always executable against the ranges
    /// [`Self::entry_slabs`] then reports. Overflow propagates with the
    /// previous range intact: the caller skips the layer loudly rather than
    /// drawing garbage.
    pub fn plan_sync(
        &mut self,
        key: LayerKey,
        content_token: u64,
        counts: [u32; SlabKind::COUNT],
    ) -> Result<SyncPlan, SlabOverflow> {
        if let Some(entry) = self.entries.get(&key) {
            let resident = !entry.poisoned
                && !entry.awaiting_rerecord
                && entry.content_token == content_token
                && entry.counts == counts
                && entry.uploaded_generation == Some(entry.slabs.generation);
            if resident {
                return Ok(SyncPlan::Clean);
            }
        }

        let mut relocated_any_kind = false;
        for kind in SlabKind::ALL {
            let index = kind.index();
            let current = self
                .allocator
                .slabs(key)
                .map(|slabs| slabs.slab(kind))
                .unwrap_or(slab::SlabRange::EMPTY);
            let range = self.allocator.realloc(key, kind, counts[index])?;
            if !range.is_empty() && range.base != current.base {
                relocated_any_kind = true;
            }
        }

        let generation = self.allocator.generation(key);
        let slabs = self
            .allocator
            .slabs(key)
            .expect("realloc above just created or updated the record");
        let (transform_slot, previous_pages, previous_awaiting, previous_token) =
            match self.entries.remove(&key) {
                Some(entry) => (
                    entry.transform_slot,
                    entry.referenced_pages,
                    entry.awaiting_rerecord,
                    entry.content_token,
                ),
                None => (self.transforms.slot_for(key), FxHashSet::default(), false, 0),
            };
        self.entries.insert(
            key,
            RegistryEntry {
                content_token,
                slabs,
                counts,
                uploaded_generation: generation,
                poisoned: false,
                // The eviction gate only clears when the layer actually
                // rebuilt: a same-token resync would otherwise re-upload the
                // very bytes whose tile ids went stale.
                awaiting_rerecord: previous_awaiting && previous_token == content_token,
                referenced_pages: previous_pages,
                last_referenced_frame: self.frame,
                transform_slot,
            },
        );

        if relocated_any_kind {
            crate::render_stats::count(COUNTER_LAYERS_REALLOCATED);
        }
        self.uploads_since_last_plan = self.uploads_since_last_plan.saturating_add(1);

        Ok(SyncPlan::UploadAllOccupied)
    }

    /// The layer's current slab snapshot, once a plan has been executed.
    pub fn entry_slabs(&self, key: LayerKey) -> Option<LayerSlabs> {
        self.entries.get(&key).map(|entry| entry.slabs)
    }

    pub fn transform_slot(&self, key: LayerKey) -> Option<u32> {
        self.entries.get(&key).map(|entry| entry.transform_slot)
    }

    /// Record the atlas pages a freshly synced layer references.
    pub fn note_referenced_pages(
        &mut self,
        key: LayerKey,
        pages: impl IntoIterator<Item = (u32, crate::AtlasTextureKind)>,
    ) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.referenced_pages.clear();
            entry.referenced_pages.extend(pages);
            entry.last_referenced_frame = self.frame;
        }
    }

    pub fn note_span_drawn_clean(&self) {
        crate::render_stats::count(COUNTER_SPANS_DRAWN_CLEAN);
    }

    pub fn note_span_skipped_awaiting_rerecord(&self) {
        crate::render_stats::count(COUNTER_SPANS_SKIPPED_EVICTED);
    }

    /// Whether spans for this key must not draw this frame (eviction pending).
    pub fn is_awaiting_rerecord(&self, key: LayerKey) -> bool {
        self.entries
            .get(&key)
            .is_some_and(|entry| entry.awaiting_rerecord)
    }

    /// Poison entries referencing any of `pages` (evicted atlas textures).
    ///
    /// Poisoned layers skip draws until their token changes via re-record —
    /// uploading would only push stale tile ids back to the GPU. Returns the
    /// keys that were poisoned, for the re-record queue.
    pub fn poison_on_evicted_pages(&mut self, pages: &[AtlasTextureId]) -> Vec<LayerKey> {
        if pages.is_empty() {
            return Vec::new();
        }
        let page_set: FxHashSet<(u32, crate::AtlasTextureKind)> = pages
            .iter()
            .map(|page| (page.index, page.kind))
            .collect();
        let mut poisoned = Vec::new();
        for (key, entry) in self.entries.iter_mut() {
            if entry.poisoned || entry.awaiting_rerecord {
                continue;
            }
            if entry
                .referenced_pages
                .iter()
                .any(|page| page_set.contains(page))
            {
                entry.poisoned = true;
                entry.awaiting_rerecord = true;
                poisoned.push(*key);
            }
        }
        if !poisoned.is_empty() {
            crate::render_stats::add(COUNTER_EVICTION_POISONED, poisoned.len() as u64);
        }
        poisoned
    }

    /// Advisory compaction passthrough: report, plan, apply.
    ///
    /// After applying, affected entries' snapshots and upload marks are
    /// rewritten so the caller's GPU block copies satisfy residency without a
    /// re-upload; correctness does not depend on any of this running.
    pub fn should_compact(&self, utilization_threshold: f32) -> bool {
        self.allocator.should_compact(utilization_threshold)
    }

    /// Whether the caller may build and apply a compaction plan this frame.
    ///
    /// Three independent gates, all pure scheduling (never correctness):
    ///
    /// - the zero-move backoff cooldown ([`Self::note_zero_move_plan`]);
    /// - upload traffic since the last evaluation — planning while layers are
    ///   being uploaded produces instantly-stale plans and steals CPU from
    ///   the upload path.
    ///
    /// Evaluating the gate consumes the upload signal: a closed gate defers
    /// planning by exactly one frame rather than parking it forever.
    pub fn compaction_gate_open(&mut self) -> bool {
        let uploaded_since_last_plan = std::mem::take(&mut self.uploads_since_last_plan) > 0;
        if self.compaction_cooldown_frames > 0 {
            self.compaction_cooldown_frames -= 1;
            crate::render_stats::count(COUNTER_COMPACTION_PLANS_DEFERRED);
            return false;
        }
        if uploaded_since_last_plan {
            crate::render_stats::count(COUNTER_COMPACTION_PLANS_DEFERRED);
            return false;
        }
        true
    }

    /// Record that a plan was built and applied zero moves. The arenas were
    /// sparse but already packed — idle-window steady state — so back off
    /// before planning again.
    pub fn note_zero_move_plan(&mut self) {
        self.zero_move_streak = self.zero_move_streak.saturating_add(1);
        let shift = self.zero_move_streak.saturating_sub(1).min(COMPACTION_BACKOFF_MAX_SHIFT);
        self.compaction_cooldown_frames = COMPACTION_BACKOFF_BASE_FRAMES << shift;
        crate::render_stats::count(COUNTER_ZERO_MOVE_PLANS);
    }

    /// Record that a plan moved something; reset the backoff schedule.
    pub fn note_moves_applied(&mut self) {
        self.zero_move_streak = 0;
        self.compaction_cooldown_frames = 0;
    }

    #[cfg(test)]
    pub(crate) fn compaction_cooldown_frames(&self) -> u32 {
        self.compaction_cooldown_frames
    }

    pub fn compaction_plan(&self) -> CompactionPlan<LayerKey> {
        self.allocator.compaction_plan()
    }

    /// Apply `plan`, returning the moves whose bytes the caller must copy on
    /// the GPU (source range, destination range, per kind).
    pub fn apply_compaction(
        &mut self,
        plan: &CompactionPlan<LayerKey>,
    ) -> Vec<(SlabKind, slab::SlabRange, slab::SlabRange)> {
        let applied_layers = self.allocator.apply_compaction(plan);
        if applied_layers == 0 {
            return Vec::new();
        }
        crate::render_stats::count(COUNTER_COMPACTIONS);

        let mut copies = Vec::new();
        for movement in &plan.moves {
            copies.push((movement.kind, movement.src, movement.dst));
        }

        for key in self.entries.keys().copied().collect::<Vec<LayerKey>>() {
            let Some(slabs) = self.allocator.slabs(key) else {
                continue;
            };
            let generation = self.allocator.generation(key);
            let entry = self.entries.get_mut(&key).expect("key came from entries");
            entry.slabs = slabs;
            // The caller copies every moved range on the GPU this frame, so
            // moved layers stay resident even though apply_compaction bumped
            // their generations.
            entry.uploaded_generation = generation;
        }

        copies
    }

    /// Arena element capacity for `kind`: what the backing GPU buffer holds.
    pub fn arena_element_capacity(&self, kind: SlabKind) -> u32 {
        self.allocator.arena_element_capacity(kind)
    }

    /// The transform table, for slot writes and bind-group offsets.
    pub fn transforms_shared(&self) -> &TransformTable {
        &self.transforms
    }

    /// Void every entry's residency mark (a slab buffer was recreated and its
    /// bytes are gone). Next sync of each layer re-uploads in full.
    pub fn invalidate_all_residency(&mut self) {
        for entry in self.entries.values_mut() {
            entry.uploaded_generation = None;
        }
    }

    /// Mark every assigned transform slot stale (the uniform buffer was
    /// recreated). Slot 0 stays zero, which is its correct content.
    pub fn mark_all_transforms_dirty(&mut self) {
        self.transforms.mark_all_dirty();
    }

    /// Set a layer's translate, reporting whether the uniform is now stale.
    pub fn set_layer_translate(&mut self, key: LayerKey, translate: [f32; 2]) -> bool {
        let changed = self.transforms.set_translate(key, translate);
        if changed {
            crate::render_stats::count(COUNTER_TRANSFORM_SLOTS_WRITTEN);
        }
        changed
    }

    /// [`TransformTable::drain_dirty_into`] passthrough into caller-owned
    /// storage, so the per-frame drain reuses the renderer's scratch vector
    /// instead of allocating.
    pub fn take_dirty_transforms_into(&mut self, out: &mut Vec<(u32, GpuLayerTransform)>) {
        self.transforms.drain_dirty_into(out);
    }

    /// Allocating variant of [`Self::take_dirty_transforms_into`], kept for
    /// the GPU-tier test harness's span preparation.
    #[cfg(test)]
    pub fn take_dirty_transforms(&mut self) -> Vec<(u32, GpuLayerTransform)> {
        let mut out = Vec::new();
        self.take_dirty_transforms_into(&mut out);
        out
    }

    fn drop_entry(&mut self, key: &LayerKey) {
        self.allocator.free_layer(*key);
        self.transforms.release(*key);
        self.entries.remove(key);
    }

    #[cfg(test)]
    pub fn live_elements(&self, kind: SlabKind) -> u64 {
        self.allocator.live_elements(kind)
    }

    #[cfg(test)]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

static OVERFLOW_WARNED: LazyLock<std::sync::atomic::AtomicBool> =
    LazyLock::new(|| std::sync::atomic::AtomicBool::new(false));

/// Report a slab overflow loudly. Overflow requires one layer claiming more
/// than two billion instances of one kind; the frame skips that layer's spans
/// (visible gap) rather than drawing wrong pixels.
pub(crate) fn report_sync_overflow(error: SlabOverflow) {
    if !OVERFLOW_WARNED.swap(true, std::sync::atomic::Ordering::AcqRel) {
        log::error!(
            "slab sync overflow in {:?}: layer cannot be placed in a u32-indexed \
             buffer; skipping its draws until a smaller rebuild lands",
            error.kind
        );
    }
    crate::render_stats::count(COUNTER_SYNC_OVERFLOWS);
}

static MISSING_RUNS_WARNED: LazyLock<std::sync::atomic::AtomicBool> =
    LazyLock::new(|| std::sync::atomic::AtomicBool::new(false));

/// Fail-loud stand-in for "missing runs / missing registry state" on a span:
/// warn once + counter + poison so the next frame re-renders legacy.
pub(crate) fn report_missing_slab_state(key: LayerKey) {
    if !MISSING_RUNS_WARNED.swap(true, std::sync::atomic::Ordering::AcqRel) {
        log::warn!(
            "slab span for {key:?} resolved against missing registry state; \
             requesting re-record"
        );
    }
    crate::render_stats::count(COUNTER_SPANS_SKIPPED_EVICTED);
    request_rerecord([key]);
}

// ---------------------------------------------------------------------
// GPU buffers.
//
// One grow-only storage buffer per [`SlabKind`], plus the transform uniform.
// Growth recreates the buffer, which orphans its contents — callers must
// treat a `true` from the ensure methods as "every layer's residency is
// void; re-upload everything", which the registry models via
// `invalidate_all_residency`.
// ---------------------------------------------------------------------

/// Byte width of one element of `kind`'s stream. Path slabs hold flattened
/// `GpuPathVertex`s (48 bytes), not path structs.
pub(crate) fn instance_stride(kind: SlabKind) -> u64 {
    match kind {
        // crate::scene::Quad
        SlabKind::Quads => 168,
        // crate::scene::Shadow
        SlabKind::Shadows => 72,
        // GpuPathVertex (renderer.rs); paths draw as vertex ranges
        SlabKind::Paths => 48,
        // crate::scene::Underline
        SlabKind::Underlines => 64,
        // crate::scene::MonochromeSprite
        SlabKind::MonoSprites => 168,
        // crate::scene::PolychromeSprite
        SlabKind::PolySprites => 96,
    }
}

const INITIAL_KIND_BUFFER_ELEMENTS: u64 = 1024;
const INITIAL_TRANSFORM_SLOTS: u32 = 128;

pub(crate) struct SlabGpuBuffers {
    kinds: [wgpu::Buffer; SlabKind::COUNT],
    transforms: wgpu::Buffer,
    /// Dynamic-offset stride of one transform slot: `max(alignment, 64)`.
    pub transform_slot_stride: u64,
}

impl SlabGpuBuffers {
    pub fn new(device: &wgpu::Device, min_uniform_offset_alignment: u32) -> Self {
        let transform_slot_stride = (min_uniform_offset_alignment as u64)
            .max(std::mem::size_of::<GpuLayerTransform>() as u64);
        let kinds = std::array::from_fn(|index| {
            let kind = SlabKind::ALL[index];
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("layer slab buffer"),
                size: INITIAL_KIND_BUFFER_ELEMENTS * instance_stride(kind),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        let transforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer slab transforms"),
            size: transform_slot_stride * INITIAL_TRANSFORM_SLOTS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        SlabGpuBuffers {
            kinds,
            transforms,
            transform_slot_stride,
        }
    }

    pub fn kind_buffer(&self, kind: SlabKind) -> &wgpu::Buffer {
        &self.kinds[kind.index()]
    }

    pub fn transforms_buffer(&self) -> &wgpu::Buffer {
        &self.transforms
    }

    /// Grow `kind`'s buffer to fit `elements`. Returns whether it was
    /// recreated (contents lost).
    pub fn ensure_kind_capacity(
        &mut self,
        device: &wgpu::Device,
        kind: SlabKind,
        elements: u32,
    ) -> bool {
        let needed = elements as u64 * instance_stride(kind);
        if self.kinds[kind.index()].size() >= needed {
            return false;
        }
        self.kinds[kind.index()] = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer slab buffer"),
            size: (needed * 2).max(INITIAL_KIND_BUFFER_ELEMENTS * instance_stride(kind)),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        true
    }

    /// Grow the transform uniform to fit `slots`. Returns whether it was
    /// recreated (all slot contents lost).
    pub fn ensure_transform_capacity(&mut self, device: &wgpu::Device, slots: u32) -> bool {
        let needed = self.transform_slot_stride * slots.max(2) as u64;
        if self.transforms.size() >= needed {
            return false;
        }
        self.transforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("layer slab transforms"),
            size: needed * 2,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        true
    }
}

// ---------------------------------------------------------------------
// Relative packing.
//
// Slab instances are stored relative to their layer's composited origin; the
// per-layer transform uniform restores window space in-shader. This is what
// makes a TRANSFORM-only change cost exactly one 64-byte uniform write
// instead of a full instance re-upload. Fragment stages undo the translate
// before comparing against mask/bounds data, so clip behaviour moves with
// the geometry.
// ---------------------------------------------------------------------

/// Rewrite `packed`'s coordinates from window space to `origin`-relative
/// space. Must be applied once, at pack time, before any upload.
pub(crate) fn make_packed_relative(
    packed: &mut crate::scene_pack::PackedLayer,
    origin: [f32; 2],
) {
    fn shift_bounds(
        bounds: &mut crate::Bounds<crate::ScaledPixels>,
        origin: [f32; 2],
    ) {
        bounds.origin.x.0 -= origin[0];
        bounds.origin.y.0 -= origin[1];
    }

    for quad in &mut packed.quads {
        shift_bounds(&mut quad.bounds, origin);
        shift_bounds(&mut quad.content_mask.bounds, origin);
    }
    for shadow in &mut packed.shadows {
        shift_bounds(&mut shadow.bounds, origin);
        shift_bounds(&mut shadow.content_mask.bounds, origin);
    }
    for underline in &mut packed.underlines {
        shift_bounds(&mut underline.bounds, origin);
        shift_bounds(&mut underline.content_mask.bounds, origin);
    }
    // A sprite's own transformation is world-space too. The shader evaluates
    // `RS * pos_relative + translation_relative`, then adds the uniform:
    // `RS * pos_absolute + translation_absolute` requires
    // `translation_relative = translation_absolute + RS * origin - origin`.
    for sprite in &mut packed.mono_sprites {
        shift_bounds(&mut sprite.bounds, origin);
        shift_bounds(&mut sprite.content_mask.bounds, origin);
        let rs = &sprite.transformation.rotation_scale;
        let translation = sprite.transformation.translation;
        sprite.transformation.translation = [
            translation[0] + rs[0][0] * origin[0] + rs[0][1] * origin[1] - origin[0],
            translation[1] + rs[1][0] * origin[0] + rs[1][1] * origin[1] - origin[1],
        ];
    }
    for sprite in &mut packed.poly_sprites {
        shift_bounds(&mut sprite.bounds, origin);
        shift_bounds(&mut sprite.content_mask.bounds, origin);
    }
    for path in &mut packed.paths {
        shift_bounds(&mut path.bounds, origin);
        shift_bounds(&mut path.content_mask.bounds, origin);
        for vertex in &mut path.vertices {
            vertex.xy_position.x.0 -= origin[0];
            vertex.xy_position.y.0 -= origin[1];
            shift_bounds(&mut vertex.content_mask.bounds, origin);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::cross::slab::MIN_CLASS;

    const KEY: LayerKey = LayerKey(42);
    const OTHER: LayerKey = LayerKey(7);

    fn counts_of(entries: &[(SlabKind, u32)]) -> [u32; SlabKind::COUNT] {
        let mut counts = [0u32; SlabKind::COUNT];
        for (kind, count) in entries {
            counts[kind.index()] = *count;
        }
        counts
    }

    #[test]
    fn first_sync_uploads_then_idle_frames_are_clean() {
        let mut registry = SlabRegistry::new();
        let counts = counts_of(&[(SlabKind::Quads, 10), (SlabKind::MonoSprites, 3)]);

        let plan = registry.plan_sync(KEY, 1, counts).unwrap();
        assert_eq!(plan, SyncPlan::UploadAllOccupied);
        let slabs = registry.entry_slabs(KEY).unwrap();
        assert_eq!(slabs.slab(SlabKind::Quads).count, 10);
        assert_eq!(slabs.slab(SlabKind::Quads).capacity, MIN_CLASS);

        // Same token, same counts: zero uploads forever after.
        for _ in 0..5 {
            registry.begin_frame();
            let plan = registry.plan_sync(KEY, 1, counts).unwrap();
            assert_eq!(plan, SyncPlan::Clean);
        }
    }

    #[test]
    fn token_change_forces_reupload_without_disturbing_other_layers() {
        let mut registry = SlabRegistry::new();
        let counts = counts_of(&[(SlabKind::Quads, 8)]);
        registry.plan_sync(KEY, 1, counts).unwrap();
        registry.plan_sync(OTHER, 1, counts).unwrap();

        let plan = registry.plan_sync(KEY, 2, counts).unwrap();
        assert_eq!(plan, SyncPlan::UploadAllOccupied);

        let other_plan = registry.plan_sync(OTHER, 1, counts).unwrap();
        assert_eq!(other_plan, SyncPlan::Clean, "unrelated layer stays clean");
    }

    #[test]
    fn count_growth_within_class_stays_put_but_reuploads_only_that_layer() {
        let mut registry = SlabRegistry::new();
        let small = counts_of(&[(SlabKind::Quads, 10)]);
        registry.plan_sync(KEY, 1, small).unwrap();
        registry.plan_sync(OTHER, 1, small).unwrap();
        let base = registry.entry_slabs(KEY).unwrap().slab(SlabKind::Quads).base;

        let grown = counts_of(&[(SlabKind::Quads, 60)]);
        let plan = registry.plan_sync(KEY, 2, grown).unwrap();
        assert_eq!(plan, SyncPlan::UploadAllOccupied);
        assert_eq!(
            registry.entry_slabs(KEY).unwrap().slab(SlabKind::Quads).base,
            base,
            "same size class must not relocate"
        );
        // OTHER untouched: still resident, next sync reads Clean.
        registry.begin_frame();
        assert_eq!(registry.plan_sync(OTHER, 1, small).unwrap(), SyncPlan::Clean);
    }

    #[test]
    fn growth_across_classes_relocates_when_space_is_pinned() {
        let mut registry = SlabRegistry::new();
        let small = counts_of(&[(SlabKind::Shadows, 10)]);
        registry.plan_sync(KEY, 1, small).unwrap();
        registry
            .plan_sync(OTHER, 1, counts_of(&[(SlabKind::Shadows, MIN_CLASS)]))
            .unwrap();
        let original_base = registry.entry_slabs(KEY).unwrap().slab(SlabKind::Shadows).base;

        let big = counts_of(&[(SlabKind::Shadows, MIN_CLASS * 2)]);
        let plan = registry.plan_sync(KEY, 2, big).unwrap();
        assert_eq!(plan, SyncPlan::UploadAllOccupied);
        assert_ne!(
            registry.entry_slabs(KEY).unwrap().slab(SlabKind::Shadows).base,
            original_base,
            "crossing a class boundary with pinned neighbours must relocate"
        );
    }

    #[test]
    fn shrink_to_zero_frees_the_range_and_later_growth_reuploads() {
        let mut registry = SlabRegistry::new();
        registry
            .plan_sync(KEY, 1, counts_of(&[(SlabKind::Underlines, 20)]))
            .unwrap();

        registry.plan_sync(KEY, 2, counts_of(&[])).unwrap();
        assert_eq!(registry.live_elements(SlabKind::Underlines), 0);

        let plan = registry
            .plan_sync(KEY, 3, counts_of(&[(SlabKind::Underlines, 4)]))
            .unwrap();
        assert_eq!(plan, SyncPlan::UploadAllOccupied);
    }

    #[test]
    fn overflow_leaves_previous_state_intact() {
        let mut registry = SlabRegistry::new();
        registry
            .plan_sync(KEY, 1, counts_of(&[(SlabKind::Quads, 10)]))
            .unwrap();

        let huge = [u32::MAX, 0, 0, 0, 0, 0];
        let error = registry.plan_sync(KEY, 2, huge).unwrap_err();
        assert_eq!(error.kind, SlabKind::Quads);

        // The old range survives; the layer keeps drawing until rebuilt.
        assert_eq!(
            registry.entry_slabs(KEY).unwrap().slab(SlabKind::Quads).count,
            10
        );
    }

    #[test]
    fn poisoned_entries_skip_until_token_changes() {
        let mut registry = SlabRegistry::new();
        registry
            .plan_sync(KEY, 1, counts_of(&[(SlabKind::PolySprites, 5)]))
            .unwrap();

        // Simulate eviction poisoning.
        {
            let entry = registry.entries.get_mut(&KEY).unwrap();
            entry.poisoned = true;
            entry.awaiting_rerecord = true;
        }

        assert!(registry.is_awaiting_rerecord(KEY));
        // Same token cannot clear the flag: stale tile ids must not re-upload.
        let plan = registry
            .plan_sync(KEY, 1, counts_of(&[(SlabKind::PolySprites, 5)]))
            .unwrap();
        assert_eq!(plan, SyncPlan::UploadAllOccupied);
        assert!(
            registry.is_awaiting_rerecord(KEY),
            "same-token resync must not clear the eviction gate"
        );

        // A re-record bumps the token, which clears everything.
        let plan = registry
            .plan_sync(KEY, 2, counts_of(&[(SlabKind::PolySprites, 5)]))
            .unwrap();
        assert_eq!(plan, SyncPlan::UploadAllOccupied);
        assert!(!registry.is_awaiting_rerecord(KEY));
    }

    #[test]
    fn eviction_poisons_exactly_the_referencing_layers() {
        let mut registry = SlabRegistry::new();
        registry
            .plan_sync(KEY, 1, counts_of(&[(SlabKind::MonoSprites, 2)]))
            .unwrap();
        registry
            .plan_sync(OTHER, 1, counts_of(&[(SlabKind::Quads, 2)]))
            .unwrap();
        registry.note_referenced_pages(KEY, [(3, crate::AtlasTextureKind::Monochrome)]);

        let page = AtlasTextureId {
            index: 3,
            kind: crate::AtlasTextureKind::Monochrome,
        };
        let poisoned = registry.poison_on_evicted_pages(&[page]);
        assert_eq!(poisoned, vec![KEY]);
        assert!(registry.is_awaiting_rerecord(KEY));
        assert!(!registry.is_awaiting_rerecord(OTHER));

        // Draining again changes nothing.
        assert!(registry.poison_on_evicted_pages(&[page]).is_empty());
    }

    #[test]
    fn gc_reclaims_long_idle_entries_and_frees_arena_space() {
        let mut registry = SlabRegistry::new();
        registry
            .plan_sync(KEY, 1, counts_of(&[(SlabKind::Quads, 30)]))
            .unwrap();
        registry.note_referenced_pages(KEY, []);
        assert!(registry.arena_element_capacity(SlabKind::Quads) > 0);

        for _ in 0..(GC_IDLE_FRAMES + 2) {
            registry.begin_frame();
        }
        assert_eq!(registry.entry_count(), 0);
        assert_eq!(registry.live_elements(SlabKind::Quads), 0);

        // Referencing again after gc starts over from a clean slate.
        let plan = registry
            .plan_sync(KEY, 9, counts_of(&[(SlabKind::Quads, 30)]))
            .unwrap();
        assert_eq!(plan, SyncPlan::UploadAllOccupied);
        assert_eq!(
            registry.entry_slabs(KEY).unwrap().slab(SlabKind::Quads).base,
            0
        );
    }

    #[test]
    fn recently_referenced_entries_survive_gc() {
        let mut registry = SlabRegistry::new();
        registry
            .plan_sync(KEY, 1, counts_of(&[(SlabKind::Quads, 30)]))
            .unwrap();
        for _ in 0..10 {
            registry.begin_frame();
            registry
                .plan_sync(KEY, 1, counts_of(&[(SlabKind::Quads, 30)]))
                .unwrap();
        }
        assert_eq!(registry.entry_count(), 1);
    }

    #[test]
    fn transform_table_assigns_reuses_and_tracks_dirty_slots() {
        let mut table = TransformTable::new();
        assert_eq!(table.slot_count(), 1, "slot 0 reserved for legacy identity");

        let first = table.slot_for(KEY);
        assert_ne!(first, 0);
        assert!(table.set_translate(KEY, [12.0, -3.0]));
        let dirty = table.drain_dirty();
        assert_eq!(dirty.len(), 1);
        assert_eq!(
            dirty[0],
            (
                first,
                GpuLayerTransform {
                    translate: [12.0, -3.0],
                    ..Default::default()
                }
            )
        );

        // Unchanged value: nothing dirty.
        assert!(!table.set_translate(KEY, [12.0, -3.0]));
        assert!(table.drain_dirty().is_empty());

        // Released slots are recycled with reset contents.
        table.release(KEY);
        let recycled = table.slot_for(OTHER);
        assert_eq!(recycled, first);
        assert_eq!(table.slot_value(recycled).translate, [0.0, 0.0]);
    }

    // -----------------------------------------------------------------
    // Upload-decision write log: the decision function drives an executor
    // that records what would be pushed to the queue, so idle/move/dirty
    // behaviour is asserted without a device.
    // -----------------------------------------------------------------

    #[derive(Debug, PartialEq, Eq)]
    enum RecordedWrite {
        Kind(SlabKind, u64, usize),
        Transform(u32),
    }

    struct WriteLog {
        writes: Vec<RecordedWrite>,
    }

    impl WriteLog {
        fn new() -> Self {
            WriteLog { writes: Vec::new() }
        }

        /// Execute one frame exactly like `WgpuRenderer::resolve_slab_spans`
        /// does: one sync decision per layer, writes per occupied kind at
        /// element-unit byte offsets, then dirty transform slots.
        fn execute_frame(&mut self, registry: &mut SlabRegistry, frames: &[FrameInput]) {
            let stride_for = |kind: SlabKind| instance_stride(kind);
            let mut synced = FxHashSet::default();
            for frame in frames {
                if synced.insert(frame.key) {
                    match registry.plan_sync(frame.key, frame.token, frame.counts) {
                        Err(_) => {}
                        Ok(SyncPlan::Clean) => registry.note_span_drawn_clean(),
                        Ok(SyncPlan::UploadAllOccupied) => {
                            let slabs = registry.entry_slabs(frame.key).unwrap();
                            for kind in SlabKind::ALL {
                                let range = slabs.slab(kind);
                                if range.is_empty() || frame.counts[kind.index()] == 0 {
                                    continue;
                                }
                                debug_assert_eq!(range.count, frame.counts[kind.index()]);
                                self.writes.push(RecordedWrite::Kind(
                                    kind,
                                    range.byte_offset(stride_for(kind)),
                                    range.count as usize * stride_for(kind) as usize,
                                ));
                            }
                        }
                    }
                    registry.set_layer_translate(frame.key, frame.origin);
                    registry.note_referenced_pages(frame.key, []);
                }
            }
            let mut dirty_transforms = Vec::new();
            registry.take_dirty_transforms_into(&mut dirty_transforms);
            for &(slot, _) in &dirty_transforms {
                self.writes.push(RecordedWrite::Transform(slot));
            }
            registry.begin_frame();
        }
    }

    struct FrameInput {
        key: LayerKey,
        token: u64,
        counts: [u32; SlabKind::COUNT],
        origin: [f32; 2],
    }

    #[test]
    fn idle_frames_record_zero_writes_and_a_move_records_one_transform_slot() {
        let mut registry = SlabRegistry::new();
        let mut log = WriteLog::new();

        let counts = counts_of(&[(SlabKind::Quads, 100), (SlabKind::Paths, 40)]);
        let first = FrameInput { key: KEY, token: 1, counts, origin: [0.0, 0.0] };
        log.execute_frame(&mut registry, &[first]);
        let initial_writes = log.writes.len();
        assert!(initial_writes >= 2, "the first upload writes quads + paths");
        log.writes.clear();

        // Idle: same token, same counts, same origin → nothing at all.
        let idle = FrameInput { key: KEY, token: 1, counts, origin: [0.0, 0.0] };
        log.execute_frame(&mut registry, &[idle]);
        assert!(
            log.writes.is_empty(),
            "idle window must issue zero write_buffer calls, got {:#?}",
            log.writes
        );

        // TRANSFORM-only move: exactly one 64-byte uniform slot, no instance
        // bytes.
        let moved = FrameInput { key: KEY, token: 1, counts, origin: [40.0, 90.0] };
        log.execute_frame(&mut registry, &[moved]);
        assert_eq!(
            log.writes,
            vec![RecordedWrite::Transform(
                registry.transform_slot(KEY).unwrap()
            )],
        );

        // And the following idle frame is silent again.
        log.writes.clear();
        let settled = FrameInput { key: KEY, token: 1, counts, origin: [40.0, 90.0] };
        log.execute_frame(&mut registry, &[settled]);
        assert!(log.writes.is_empty());
    }

    #[test]
    fn a_dirty_layer_uploads_only_its_own_slab() {
        let mut registry = SlabRegistry::new();
        let mut log = WriteLog::new();
        let counts = counts_of(&[(SlabKind::Quads, 20)]);
        let base = FrameInput { key: KEY, token: 1, counts, origin: [0.0, 0.0] };
        let neighbour = FrameInput { key: OTHER, token: 1, counts, origin: [0.0, 0.0] };
        log.execute_frame(&mut registry, &[base, neighbour]);
        log.writes.clear();

        // KEY re-rendered; OTHER did not. Only KEY's range may move.
        let dirty = FrameInput { key: KEY, token: 2, counts, origin: [0.0, 0.0] };
        log.execute_frame(&mut registry, &[dirty]);

        assert_eq!(log.writes.len(), 1);
        match log.writes[0] {
            RecordedWrite::Kind(SlabKind::Quads, _, _) => {}
            ref other => panic!("expected a quads upload, got {other:?}"),
        }
    }

    #[test]
    fn uploads_sit_at_element_unit_offsets() {
        let mut registry = SlabRegistry::new();
        let mut log = WriteLog::new();
        // Two layers so the second lands at a nonzero class boundary.
        let counts = counts_of(&[(SlabKind::Quads, MIN_CLASS)]);
        let a = FrameInput { key: KEY, token: 1, counts, origin: [0.0; 2] };
        let b = FrameInput { key: OTHER, token: 1, counts, origin: [0.0; 2] };
        log.execute_frame(&mut registry, &[a, b]);

        let other_base = registry.entry_slabs(OTHER).unwrap().slab(SlabKind::Quads).base;
        assert!(other_base >= MIN_CLASS);
        let stride = instance_stride(SlabKind::Quads);
        assert_eq!(
            log.writes[1],
            RecordedWrite::Kind(SlabKind::Quads, other_base as u64 * stride, MIN_CLASS as usize * stride as usize),
        );
    }

    #[test]
    fn compaction_preserves_residency_for_copied_ranges() {
        let mut registry = SlabRegistry::new();
        // Big enough to clear should_compact's minimum-arena floor (16 Ki
        // elements): three 8 Ki-class reservations, then free the middle one.
        let big = counts_of(&[(SlabKind::Quads, MIN_CLASS * 128)]);
        registry.plan_sync(KEY, 1, big).unwrap();
        registry
            .plan_sync(OTHER, 1, counts_of(&[(SlabKind::Quads, MIN_CLASS * 128)]))
            .unwrap();
        registry.plan_sync(LayerKey(99), 1, big).unwrap();
        // Free the middle layer through a shrink-to-zero to punch a hole.
        registry.plan_sync(OTHER, 2, counts_of(&[])).unwrap();

        assert!(registry.should_compact(0.7));
        let plan = registry.compaction_plan();
        let copies = registry.apply_compaction(&plan);
        assert_eq!(copies.len(), 1);
        let (kind, src, dst) = copies[0];
        assert_eq!(kind, SlabKind::Quads);
        assert_eq!(dst.base, MIN_CLASS * 128);
        assert_eq!(src.count, dst.count);

        // Residency survived: the copied layer reads Clean despite the
        // allocator bumping generations.
        let moved_slabs = registry.entry_slabs(LayerKey(99)).unwrap();
        assert_eq!(moved_slabs.slab(SlabKind::Quads).base, dst.base);
        assert_eq!(registry.plan_sync(LayerKey(99), 1, big).unwrap(), SyncPlan::Clean);
    }

    #[test]
    fn re_record_requests_round_trip_through_the_global_queue() {
        request_rerecord([KEY, OTHER]);
        let drained = take_rerecord_requests();
        assert!(drained.contains(&KEY) && drained.contains(&OTHER));
        assert!(take_rerecord_requests().is_empty(), "drain must be total");
    }

    #[test]
    fn gpu_layer_transform_is_64_bytes_and_defaults_to_identity() {
        assert_eq!(std::mem::size_of::<GpuLayerTransform>(), 64);
        let default = GpuLayerTransform::default();
        assert_eq!(default.translate, [0.0, 0.0]);
    }

    // -----------------------------------------------------------------
    // Compaction scheduling: the gate is pure scheduling, so these drive
    // only decision state — no allocator traffic needed to prove that a
    // zero-move plan backs off exponentially and upload traffic defers
    // planning.
    // -----------------------------------------------------------------

    fn big_counts() -> [u32; SlabKind::COUNT] {
        counts_of(&[(SlabKind::Quads, MIN_CLASS * 128)])
    }

    #[test]
    fn zero_move_plans_engage_an_exponential_backoff() {
        let mut registry = SlabRegistry::new();
        registry.plan_sync(KEY, 1, big_counts()).unwrap();

        // The setup upload defers planning once; consume that deferral.
        assert!(!registry.compaction_gate_open(), "upload defers planning");
        assert!(registry.compaction_gate_open(), "next evaluation opens");

        registry.note_zero_move_plan();
        assert_eq!(registry.compaction_cooldown_frames(), 1);

        assert!(!registry.compaction_gate_open(), "cooldown suppresses");
        assert!(
            registry.compaction_gate_open(),
            "a one-frame cooldown expires after one gated evaluation"
        );

        // Each additional zero-move plan doubles the suppression window.
        registry.note_zero_move_plan();
        assert_eq!(registry.compaction_cooldown_frames(), 2);
        for _ in 0..2 {
            assert!(!registry.compaction_gate_open());
        }
        assert!(registry.compaction_gate_open());

        registry.note_zero_move_plan();
        assert_eq!(registry.compaction_cooldown_frames(), 4);

        for _ in 0..64 {
            registry.note_zero_move_plan();
        }
        assert_eq!(
            registry.compaction_cooldown_frames(),
            compaction_backoff_cap_frames(),
            "the backoff saturates instead of overflowing"
        );
    }

    #[test]
    fn applied_moves_reset_the_backoff() {
        let mut registry = SlabRegistry::new();
        registry.plan_sync(KEY, 1, big_counts()).unwrap();
        // Drain the setup upload's deferral so the gate state is purely the
        // backoff under test.
        registry.compaction_gate_open();
        registry.compaction_gate_open();

        registry.note_zero_move_plan();
        registry.note_zero_move_plan();
        assert!(registry.compaction_cooldown_frames() > 0);

        registry.note_moves_applied();
        assert_eq!(registry.compaction_cooldown_frames(), 0);
        assert!(
            registry.compaction_gate_open(),
            "a plan that moved something reopens planning immediately"
        );
    }

    #[test]
    fn uploads_since_the_last_evaluation_defer_planning_one_frame() {
        let mut registry = SlabRegistry::new();
        registry.plan_sync(KEY, 1, big_counts()).unwrap();

        let open = registry.compaction_gate_open();
        assert!(!open, "an upload since the last evaluation closes the gate");
        assert!(
            registry.compaction_gate_open(),
            "the deferral is consumed: the next evaluation opens again"
        );
    }

    #[test]
    fn dirty_drain_reuses_caller_storage_without_reallocation() {
        let mut table = TransformTable::new();
        let slot = table.slot_for(KEY);
        table.set_translate(KEY, [1.0, 2.0]);

        let mut drained: Vec<(u32, GpuLayerTransform)> = Vec::new();
        table.drain_dirty_into(&mut drained);
        assert_eq!(drained.len(), 1);
        let pointer = drained.as_ptr();
        let capacity = drained.capacity();

        // A second drain after more dirt lands must neither reallocate nor
        // grow: same storage, same contents discipline (sorted by slot).
        let other = table.slot_for(OTHER);
        table.set_translate(KEY, [3.0, 4.0]);
        table.set_translate(OTHER, [5.0, 6.0]);
        table.drain_dirty_into(&mut drained);
        assert_eq!(drained.as_ptr(), pointer, "capacity must be reused");
        assert!(drained.capacity() >= capacity);
        assert_eq!(
            drained.iter().map(|&(slot, _)| slot).collect::<Vec<u32>>(),
            vec![slot.min(other), slot.max(other)],
            "drained slots stay sorted ascending"
        );
        assert!(table.dirty.is_empty(), "the drain must be total");
    }
}
