//! Indirect draw issuance + coalescing (today's `OpenSlabRun` logic in
//! `src/renderer.rs`). See docs/gpu-native-architecture.md §5.3, §3.5.
//!
//! # The whole of §8's Phase 4 gate lives in this file
//!
//! > A clean window's CPU-side draw-issuing work is O(layer slots), independent
//! > of resident primitive count.
//!
//! Every function below is written so that claim is *structurally* true rather
//! than true by measurement luck: nothing here reads a primitive, a record, an
//! upload, or a count. [`SlotBasePlan`] is built from a
//! `wgpui_core::indirect::SlotTable`, which is built from each layer's
//! `SlabRange`, and issuing is a loop over slots whose body is a constant
//! number of calls. [`DrawStats`] then makes it *measurable* rather than
//! asserted, in the style `render_stats` established in the legacy backend.
//!
//! # The one place the claim is not O(layer slots) flat
//!
//! [`issue_sprites`] multiplies the sequence by the number of live atlas pages
//! of its own kind, because a bind group cannot change inside a draw call and a
//! sprite's page decides its texture. That is the same reason
//! [`issue_composites`] is per entry even where multi-draw is available, and it
//! is a property of sampling textures rather than a shortcoming of the mode.
//! [`DrawStats::atlas_pages_bound`] reports the multiplier rather than leaving
//! it to be inferred from a draw count.
//!
//! The counter that carries the claim is [`DrawStats::instances_known_to_cpu`].
//! On every indirect path it is `None` — not zero, not unknown-but-guessable:
//! the CPU issued the draws without the number existing on its side of the
//! bus. §5.3's own wording is "the GPU decides how much work each indirect call
//! expands to, including 'none,' without the CPU ever finding out the count,"
//! and an `Option` is what makes that a thing a test can assert.
//!
//! # Coalescing, and what happened to it
//!
//! The legacy renderer merges byte-contiguous same-layer/same-kind runs into
//! one `pass.draw` (`OpenSlabRun`, `renderer.rs:1690-1723`), which is a real
//! optimisation *because* its draw ranges are CPU-computed: it can see that two
//! ranges abut. Here it is neither possible nor needed. Not possible, because
//! whether two slots' live instances abut is a fact about the GPU's compaction
//! that the CPU deliberately does not have. Not needed, because the sequence is
//! already one call per slot, which is what coalescing was reducing *to*, and
//! because [`DrawMode::MultiDrawIndirectCount`] collapses the whole kind into
//! one call — further than coalescing ever got.
//!
//! # The fallback is a first-class path, not a patch
//!
//! §9's risk table asks for exactly that. [`DrawMode::CpuReadback`] reads the
//! argument records back and issues ordinary draws, and it is the *reference*
//! arm of this crate's own tests: `tests/indirect_draw.rs` renders every mode
//! and compares framebuffers against it. A fallback nothing exercises is a
//! fallback that does not work, and this one is exercised on every test run on
//! hardware that has the features to skip it.

use wgpui_core::indirect::{DRAW_INDIRECT_ARGS_STRIDE, DrawIndirectArgs, DrawSlot, FirstInstance};

use crate::render::compute::indirect_args_pass::IndirectArgsBuffers;
use crate::render::device::IndirectSupport;
use crate::render::pipelines::{
    BackdropPipeline, CompositePipeline, MonoSpritePipeline, PathPipeline, PolySpritePipeline,
    QuadPipeline, ShadowPipeline, UnderlinePipeline, slot_base_bind_group,
};
use crate::render::readback::{ReadbackError, StagingReader};
use crate::render::textures::external_surface::CompositePlan;

/// How the fixed draw sequence reaches the device.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DrawMode {
    /// One `draw_indirect` per (layer, kind) slot, with the slot's base index
    /// supplied through a dynamic uniform offset and `first_instance` left at
    /// zero.
    ///
    /// The default, and the only one that needs no device feature at all. It is
    /// what WebGPU permits and it cannot produce the input README's "Custom
    /// Device Gotcha" describes, because it never writes a nonzero
    /// `firstInstance`.
    PerSlotIndirect,
    /// One `multi_draw_indirect` per kind, over every slot's record.
    ///
    /// Requires `INDIRECT_FIRST_INSTANCE` (each record addresses its own
    /// instance range) and `MULTI_DRAW_INDIRECT_COUNT` (without it wgpu 30
    /// emulates the call as a CPU-side loop of `draw_indirect`, which saves the
    /// CPU nothing — see `render/device.rs`).
    MultiDrawIndirect,
    /// One `multi_draw_indirect_count` per kind, over the *packed* records,
    /// with the record count itself read from a GPU buffer.
    ///
    /// The furthest the CPU's involvement goes: it issues one call per kind and
    /// does not know how many entries that call will find, let alone how many
    /// instances each draws.
    MultiDrawIndirectCount,
    /// Read the records back and issue ordinary draws.
    ///
    /// §5.3's fallback "for the macOS best-effort case and for WASM, which are
    /// the same fallback path … and therefore one piece of work, not two." Also
    /// this crate's reference arm — see this module's doc.
    CpuReadback,
}

impl DrawMode {
    /// The most capable mode `support` allows.
    pub const fn best_available(support: IndirectSupport) -> DrawMode {
        if support.supports_native_multi_draw() {
            DrawMode::MultiDrawIndirectCount
        } else {
            DrawMode::PerSlotIndirect
        }
    }

    /// Whether a device with `support` can take this mode.
    ///
    /// [`DrawMode::PerSlotIndirect`] and [`DrawMode::CpuReadback`] are always
    /// available: the first needs no feature and the second issues no indirect
    /// call at all.
    pub const fn is_available(self, support: IndirectSupport) -> bool {
        match self {
            DrawMode::PerSlotIndirect | DrawMode::CpuReadback => true,
            DrawMode::MultiDrawIndirect | DrawMode::MultiDrawIndirectCount => {
                support.supports_native_multi_draw()
            }
        }
    }

    /// Where this mode needs each record's base index to live.
    ///
    /// The one thing the compute pass has to know about the draw path, and the
    /// reason it is a parameter rather than a constant.
    pub const fn first_instance(self) -> FirstInstance {
        match self {
            DrawMode::MultiDrawIndirect | DrawMode::MultiDrawIndirectCount => {
                FirstInstance::SlotBase
            }
            DrawMode::PerSlotIndirect | DrawMode::CpuReadback => FirstInstance::Zero,
        }
    }

    /// Whether this mode reads the argument records back before drawing.
    pub const fn reads_back(self) -> bool {
        matches!(self, DrawMode::CpuReadback)
    }

    /// A short name for a report or a benchmark row.
    pub const fn name(self) -> &'static str {
        match self {
            DrawMode::PerSlotIndirect => "per-slot draw_indirect",
            DrawMode::MultiDrawIndirect => "multi_draw_indirect",
            DrawMode::MultiDrawIndirectCount => "multi_draw_indirect_count",
            DrawMode::CpuReadback => "CPU readback + direct draw",
        }
    }

    /// Every mode, for a test or benchmark that sweeps them.
    pub const ALL: [DrawMode; 4] = [
        DrawMode::PerSlotIndirect,
        DrawMode::MultiDrawIndirect,
        DrawMode::MultiDrawIndirectCount,
        DrawMode::CpuReadback,
    ];
}

/// What issuing one frame's draws actually cost the CPU.
///
/// R-N's `render_stats` in the shape §8's Phase 4 gate needs. Every field is a
/// count of something the CPU did, so the gate — "O(layer slots), independent
/// of resident primitive count" — is read off two runs of the same scene at
/// different primitive counts rather than off a stopwatch.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DrawStats {
    /// Entries in the fixed sequence — one per (layer, kind) slot.
    pub slots_visited: u32,
    /// Draw calls the CPU actually issued.
    pub draw_calls_issued: u32,
    /// Bind groups the CPU set.
    pub bind_group_binds: u32,
    /// Slots the CPU skipped because it had read their record and found it
    /// empty. Nonzero only on [`DrawMode::CpuReadback`]; on every indirect path
    /// the CPU cannot know, so it does not skip.
    pub slots_skipped: u32,
    /// **The gate's own counter.** How many instances the CPU knew it was
    /// drawing, or `None` when it issued the draws without the number ever
    /// crossing to its side.
    pub instances_known_to_cpu: Option<u32>,
    /// Words the CPU read back off the device to issue these draws.
    pub readback_words: u32,
    /// Composite entries considered this frame (§5.5).
    pub composite_entries_visited: u32,
    /// Composite entries the layer tier dropped, which cost no bind group, no
    /// texture fetch, and no draw call.
    pub composite_entries_culled: u32,
    /// Composite entries whose producer had nothing ready. Not an error: an
    /// external surface that has never presented, or a boundary swept out of
    /// the texture pool.
    pub composite_entries_unavailable: u32,
    /// Composite draw calls issued.
    pub composite_draws_issued: u32,
    /// Atlas pages the sprite passes bound, one bind group each.
    ///
    /// The number [`issue_sprites`] multiplies its slot sequence by, summed over
    /// both sprite pipelines, and therefore the honest form of "this pipeline's
    /// CPU cost is O(layer slots × live pages of its own kind)" rather than
    /// O(layer slots) flat.
    pub atlas_pages_bound: u32,
    /// Sprite draw calls issued, across both the monochrome (text) and
    /// polychrome (image) passes.
    ///
    /// Merged rather than split per pipeline, and named for what it counts
    /// rather than for the first pipeline that happened to need it. The two
    /// passes are the same sequence over different arenas — a split counter
    /// would suggest a difference in *kind* where there is only a difference in
    /// which texture is bound, and a test that needs to tell them apart builds a
    /// scene holding one of them.
    pub sprite_draws_issued: u32,
    /// Sprite slots that could not be issued at all, because no atlas page of
    /// the right kind was available to bind.
    ///
    /// The same idea as [`DrawStats::composite_entries_unavailable`] and not an
    /// error: a window whose text has not been rasterised yet — or whose images
    /// have not decoded yet — has sprite slots and no texture to sample them
    /// from, and there is no such thing as a draw call without a bound texture.
    /// Counted rather than silently dropped so that `slots_skipped +
    /// draw_calls_issued + sprite_slots_unavailable` still accounts for every
    /// slot the frame's fixed sequence named.
    pub sprite_slots_unavailable: u32,
    /// Path vertices issued through the direct variable-size path stream.
    pub path_vertices_issued: u32,
    /// Path layer draws issued.
    pub path_draws_issued: u32,
    /// Backdrop-filter layer draws issued after the snapshot pass.
    pub backdrop_filters_drawn: u32,
}

impl DrawStats {
    /// Fold another frame's or another pass's counts in.
    pub fn merge(&mut self, other: DrawStats) {
        self.slots_visited += other.slots_visited;
        self.draw_calls_issued += other.draw_calls_issued;
        self.bind_group_binds += other.bind_group_binds;
        self.slots_skipped += other.slots_skipped;
        self.readback_words += other.readback_words;
        self.composite_entries_visited += other.composite_entries_visited;
        self.composite_entries_culled += other.composite_entries_culled;
        self.composite_entries_unavailable += other.composite_entries_unavailable;
        self.composite_draws_issued += other.composite_draws_issued;
        self.atlas_pages_bound += other.atlas_pages_bound;
        self.sprite_draws_issued += other.sprite_draws_issued;
        self.sprite_slots_unavailable += other.sprite_slots_unavailable;
        self.path_vertices_issued += other.path_vertices_issued;
        self.path_draws_issued += other.path_draws_issued;
        self.backdrop_filters_drawn += other.backdrop_filters_drawn;
        self.instances_known_to_cpu = match (self.instances_known_to_cpu, other.instances_known_to_cpu)
        {
            // Unknown is contagious on purpose: a frame that took one indirect
            // path anywhere did not learn its own instance count, and reporting
            // the sum of the parts it *did* learn would be a smaller lie than
            // reporting zero but a lie all the same.
            (Some(left), Some(right)) => Some(left + right),
            _ => None,
        };
    }
}

/// The argument records the CPU has, which on every indirect path is none.
#[derive(Clone, Debug, Default)]
pub struct ResolvedArgs {
    records: Vec<DrawIndirectArgs>,
    words_read: u32,
}

impl ResolvedArgs {
    /// The records, empty unless the mode read them back.
    pub fn records(&self) -> &[DrawIndirectArgs] {
        &self.records
    }

    /// Words read off the device to produce this.
    pub fn words_read(&self) -> u32 {
        self.words_read
    }

    /// Read the records back if `mode` needs them, and otherwise do nothing.
    ///
    /// Called before the render pass begins, because a readback submits its own
    /// encoder and blocks.
    ///
    /// # What that costs, measured
    ///
    /// `examples/phase4_draw_issuance_bench.rs` prices this column beside the
    /// draw-issuing one, and the gap is not small. On an RTX 4060 (Vulkan,
    /// 561.03) the fallback's readback ran **446µs to 6.40ms**, against
    /// 0.3–18µs of actual draw issuing — three to four digits of difference.
    ///
    /// It is not the 2KB of argument records that costs it, and — the part
    /// worth stating precisely, because the obvious reading is wrong — it is
    /// not a function of the slot count either. Both benchmark sweeps make
    /// that visible: at a *fixed* 8 slots the readback still climbs 446µs →
    /// 1.72ms as the scene's primitive count rises, and at a fixed primitive
    /// count it climbs 853µs → 6.40ms as the layer count (and so the dispatch
    /// count) rises. `Device::poll(wait_indefinitely)` waits for *everything
    /// already submitted*, so reading the arguments back also waits for the
    /// compute dispatches that wrote them and for the previous frame's
    /// rendering to drain. The fallback does not merely add a copy; it
    /// serializes the CPU against the whole frame's GPU work, once per frame,
    /// and therefore costs whatever that frame's GPU work costs.
    ///
    /// That is worth stating plainly rather than leaving as a footnote, because
    /// §5.3 describes this path as the WASM path and the macOS best-effort
    /// path without pricing it. It is a correct path and a slow one, and a
    /// device that has `draw_indirect` at all — which is every WebGPU device —
    /// should be taking [`DrawMode::PerSlotIndirect`] instead.
    pub fn resolve(
        mode: DrawMode,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffers: &IndirectArgsBuffers,
        slot_count: u32,
        reader: &mut StagingReader,
    ) -> Result<ResolvedArgs, ReadbackError> {
        if !mode.reads_back() || slot_count == 0 {
            return Ok(ResolvedArgs::default());
        }
        let mut words = Vec::new();
        reader.read_u32(
            device,
            queue,
            &buffers.args,
            slot_count as usize * 4,
            &mut words,
        )?;
        Ok(ResolvedArgs {
            words_read: u32::try_from(words.len()).unwrap_or(u32::MAX),
            records: wgpui_core::indirect::decode_args(&words),
        })
    }
}

/// The per-slot resources one kind's fixed draw sequence needs, built once per
/// slot-table change rather than per frame.
///
/// "Per slot-table change" is doing real work in that sentence: the table is a
/// function of which layers exist and what they reserved, so it survives every
/// frame in which content changes but residency does not — which is most of
/// them.
///
/// Phase 4 called this `QuadDrawPlan`; Phase 5.6 renamed it when the glyph
/// pipeline needed exactly the same thing over a different arena. Nothing about
/// it was ever quad-specific — the slot base is `wgpui_core::indirect`'s notion,
/// not a shader's — and the two pipelines now share one implementation rather
/// than two copies that could drift.
pub struct SlotBasePlan {
    slots: Vec<DrawSlot>,
    bind_group: wgpu::BindGroup,
    stride: u32,
    /// Offset of the entry holding zero, which the multi-draw modes bind
    /// because their records carry the base themselves.
    zero_offset: u32,
}

impl SlotBasePlan {
    /// Build the plan for one kind's slots against `layout`, which must be a
    /// pipeline's slot-base bind group layout.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        stride: u32,
        slots: &[DrawSlot],
    ) -> SlotBasePlan {
        // One entry per slot, plus a trailing zero entry for the multi-draw
        // modes.
        let entries = slots.len() + 1;
        let size = u64::from(stride) * entries as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slot bases"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut bytes = vec![0u8; size as usize];
        for (index, slot) in slots.iter().enumerate() {
            let offset = index * stride as usize;
            if let Some(word) = bytes.get_mut(offset..offset + 4) {
                word.copy_from_slice(&slot.base.to_le_bytes());
            }
        }
        queue.write_buffer(&buffer, 0, &bytes);
        let bind_group = slot_base_bind_group(device, layout, &buffer);
        SlotBasePlan {
            slots: slots.to_vec(),
            bind_group,
            stride,
            zero_offset: u32::try_from(slots.len()).unwrap_or(0) * stride,
        }
    }

    /// Build the plan for the shadow pipeline's slots.
    pub fn for_shadows(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &ShadowPipeline,
        slots: &[DrawSlot],
    ) -> SlotBasePlan {
        SlotBasePlan::new(
            device,
            queue,
            &pipeline.slot_layout,
            pipeline.slot_stride,
            slots,
        )
    }

    /// Build the plan for the quad pipeline's slots.
    pub fn for_quads(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &QuadPipeline,
        slots: &[DrawSlot],
    ) -> SlotBasePlan {
        SlotBasePlan::new(
            device,
            queue,
            &pipeline.slot_layout,
            pipeline.slot_stride,
            slots,
        )
    }

    /// Build the plan for the underline pipeline's slots.
    pub fn for_underlines(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &UnderlinePipeline,
        slots: &[DrawSlot],
    ) -> SlotBasePlan {
        SlotBasePlan::new(
            device,
            queue,
            &pipeline.slot_layout,
            pipeline.slot_stride,
            slots,
        )
    }

    /// Build the plan for the mono-sprite pipeline's slots.
    pub fn for_glyphs(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &MonoSpritePipeline,
        slots: &[DrawSlot],
    ) -> SlotBasePlan {
        SlotBasePlan::new(
            device,
            queue,
            &pipeline.slot_layout,
            pipeline.slot_stride,
            slots,
        )
    }

    /// Build the plan for the poly-sprite pipeline's slots.
    pub fn for_poly_sprites(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &PolySpritePipeline,
        slots: &[DrawSlot],
    ) -> SlotBasePlan {
        SlotBasePlan::new(
            device,
            queue,
            &pipeline.slot_layout,
            pipeline.slot_stride,
            slots,
        )
    }

    /// Build the plan for flattened path streams.
    pub fn for_paths(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &PathPipeline,
        slots: &[DrawSlot],
    ) -> SlotBasePlan {
        SlotBasePlan::new(device, queue, &pipeline.slot_layout, pipeline.slot_stride, slots)
    }

    /// Build the plan for backdrop filters.
    pub fn for_backdrop_filters(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &BackdropPipeline,
        slots: &[DrawSlot],
    ) -> SlotBasePlan {
        SlotBasePlan::new(device, queue, &pipeline.slot_layout, pipeline.slot_stride, slots)
    }

    /// Entries in the fixed sequence.
    pub fn slot_count(&self) -> u32 {
        u32::try_from(self.slots.len()).unwrap_or(u32::MAX)
    }

    /// The slots this plan draws.
    pub fn slots(&self) -> &[DrawSlot] {
        &self.slots
    }

    /// The bind group holding slot bases.
    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Dynamic-uniform stride.
    pub(crate) fn stride(&self) -> u32 {
        self.stride
    }
}

/// Issue one texture-free instanced kind's fixed draw sequence.
///
/// `frame_group` is the pipeline's `frame_bind_group` output for this frame's
/// globals, arena, and indirection buffer; `args` holds the records the compute
/// pass wrote. Nothing else is read, and in particular nothing about the
/// scene's contents is.
///
/// **One function, two pipelines**, the same finding [`issue_sprites`] recorded
/// in Phase 6.2 and for the same reason: Phase 6.3's shadow pass needed this
/// body unchanged — same bind group indices, same dynamic offsets, same four
/// modes — so it took the quad name off and now takes a
/// `&wgpu::RenderPipeline` rather than one pipeline struct, because that is
/// genuinely all it reads out of either. Taking [`QuadPipeline`] would let the
/// function *look* per-kind while being per-kind in nothing but its signature.
pub fn issue_instanced(
    pass: &mut wgpu::RenderPass<'_>,
    pipeline: &wgpu::RenderPipeline,
    plan: &SlotBasePlan,
    frame_group: &wgpu::BindGroup,
    args: &IndirectArgsBuffers,
    mode: DrawMode,
    resolved: &ResolvedArgs,
) -> DrawStats {
    let mut stats = DrawStats {
        slots_visited: plan.slot_count(),
        readback_words: resolved.words_read(),
        instances_known_to_cpu: if mode.reads_back() { Some(0) } else { None },
        ..DrawStats::default()
    };
    if plan.slots.is_empty() {
        return stats;
    }

    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, frame_group, &[]);
    stats.bind_group_binds += 1;

    match mode {
        DrawMode::MultiDrawIndirectCount => {
            pass.set_bind_group(1, &plan.bind_group, &[plan.zero_offset]);
            stats.bind_group_binds += 1;
            pass.multi_draw_indirect_count(
                &args.packed_args,
                0,
                &args.draw_count,
                0,
                plan.slot_count(),
            );
            stats.draw_calls_issued += 1;
        }
        DrawMode::MultiDrawIndirect => {
            pass.set_bind_group(1, &plan.bind_group, &[plan.zero_offset]);
            stats.bind_group_binds += 1;
            pass.multi_draw_indirect(&args.args, 0, plan.slot_count());
            stats.draw_calls_issued += 1;
        }
        DrawMode::PerSlotIndirect => {
            for index in 0..plan.slots.len() {
                pass.set_bind_group(1, &plan.bind_group, &[index as u32 * plan.stride]);
                stats.bind_group_binds += 1;
                pass.draw_indirect(&args.args, index as u64 * DRAW_INDIRECT_ARGS_STRIDE as u64);
                stats.draw_calls_issued += 1;
            }
        }
        DrawMode::CpuReadback => {
            let mut known = 0u32;
            for (index, record) in resolved.records().iter().enumerate() {
                if record.is_empty() {
                    // The one thing an indirect path cannot do, and the reason
                    // this mode's `slots_skipped` is the only nonzero one: the
                    // CPU has the count, so it can decline to issue the call.
                    stats.slots_skipped += 1;
                    continue;
                }
                pass.set_bind_group(1, &plan.bind_group, &[index as u32 * plan.stride]);
                stats.bind_group_binds += 1;
                pass.draw(0..record.vertex_count, 0..record.instance_count);
                stats.draw_calls_issued += 1;
                known += record.instance_count;
            }
            stats.instances_known_to_cpu = Some(known);
        }
    }
    stats
}

/// Issue flattened path streams. A path record owns a variable number of
/// vertices, so it cannot use the fixed instance-count contract used by the
/// rectangle kinds; the slot base still comes from the retained scene and the
/// draw itself remains O(layer slots).
pub fn issue_paths(
    pass: &mut wgpu::RenderPass<'_>,
    pipeline: &PathPipeline,
    plan: &SlotBasePlan,
    frame_group: &wgpu::BindGroup,
) -> DrawStats {
    let mut stats = DrawStats {
        instances_known_to_cpu: Some(0),
        ..DrawStats::default()
    };
    if plan.slots().is_empty() {
        return stats;
    }
    pass.set_pipeline(&pipeline.pipeline);
    pass.set_bind_group(0, frame_group, &[]);
    stats.bind_group_binds += 1;
    for (index, slot) in plan.slots().iter().enumerate() {
        stats.slots_visited += 1;
        if slot.count == 0 {
            stats.slots_skipped += 1;
            continue;
        }
        pass.set_bind_group(1, plan.bind_group(), &[index as u32 * plan.stride()]);
        pass.draw(0..slot.count, 0..1);
        stats.bind_group_binds += 1;
        stats.draw_calls_issued += 1;
        stats.path_draws_issued += 1;
        stats.path_vertices_issued += slot.count;
        stats.instances_known_to_cpu = stats.instances_known_to_cpu.map(|count| count + 1);
    }
    stats
}

/// Issue backdrop filters over the framebuffer snapshot. This is deliberately
/// a direct per-layer draw: the filter pass is separated from the base pass by
/// the resource copy, and a direct draw keeps that ordering explicit.
pub fn issue_backdrop_filters(
    pass: &mut wgpu::RenderPass<'_>,
    pipeline: &BackdropPipeline,
    plan: &SlotBasePlan,
    frame_group: &wgpu::BindGroup,
    texture_group: &wgpu::BindGroup,
) -> DrawStats {
    let mut stats = DrawStats {
        instances_known_to_cpu: Some(0),
        ..DrawStats::default()
    };
    if plan.slots().is_empty() {
        return stats;
    }
    pass.set_pipeline(&pipeline.pipeline);
    pass.set_bind_group(0, frame_group, &[]);
    pass.set_bind_group(2, texture_group, &[]);
    stats.bind_group_binds += 2;
    for (index, slot) in plan.slots().iter().enumerate() {
        stats.slots_visited += 1;
        if slot.count == 0 {
            stats.slots_skipped += 1;
            continue;
        }
        pass.set_bind_group(1, plan.bind_group(), &[index as u32 * plan.stride()]);
        pass.draw(0..4, 0..slot.count);
        stats.bind_group_binds += 1;
        stats.draw_calls_issued += 1;
        stats.backdrop_filters_drawn += slot.count;
        stats.instances_known_to_cpu = stats.instances_known_to_cpu.map(|count| count + slot.count);
    }
    stats
}

/// The atlas pages one sprite pass will bind, in ascending page order.
///
/// Built per frame rather than per slot-table change, because the atlas is what
/// changes: a frame that rasterised a glyph or decoded an image into a new page
/// has a new page to bind, and a frame that evicted one has a bind group
/// referencing a texture that no longer exists. Paired with the [`SlotBasePlan`]
/// rather than folded into it for exactly that reason — the two have different
/// lifetimes, and merging them would rebuild the slot bases every time a glyph
/// was rasterised.
pub struct SpriteDraw<'a> {
    /// The pipeline this pass sets.
    ///
    /// A `&wgpu::RenderPipeline` and not one of the two pipeline structs,
    /// because that is genuinely all [`issue_sprites`] reads out of either of
    /// them. Taking the struct would let the function *look* per-kind while
    /// being per-kind in nothing but its signature.
    pub pipeline: &'a wgpu::RenderPipeline,
    /// The per-slot bases, held across frames.
    pub plan: &'a SlotBasePlan,
    /// One bind group per live page of this pipeline's own atlas kind: its
    /// index and its texture.
    pub pages: &'a [wgpu::BindGroup],
}

/// Issue one atlas-sampling kind's fixed draw sequence, once per bound page.
///
/// The sequence within a page is byte-for-byte [`issue_quads`]': the same
/// argument buffer, the same dynamic slot-base offsets, the same four modes. The
/// outer loop over `draw.pages` is the whole of the difference, and it exists
/// because both sprite shaders filter by the bound page rather than looking a
/// texture up per instance — see `mono_sprites.wgsl`'s header for why a lookup
/// is not available on a device without binding arrays.
///
/// A sprite whose page is not the bound one collapses to a degenerate triangle
/// strip, so drawing the same slot once per page is correct and not merely
/// harmless: each sprite is rasterised into the framebuffer exactly once, by the
/// pass that bound its own page.
///
/// **One function, two pipelines.** Phase 5.6 wrote this as `issue_glyphs` and
/// Phase 6.2 found that the polychrome pass needed it unchanged — same bind
/// group indices, same page loop, same mode handling — so it took the glyph name
/// off rather than copying the body. That the second sprite kind cost zero lines
/// here is the point; a duplicate would have hidden it.
///
/// **The CPU still never learns an instance count.** It issues the same record
/// `pages.len()` times and does not read it, which is why
/// [`DrawStats::instances_known_to_cpu`] is `None` on every indirect path here
/// too.
pub fn issue_sprites(
    pass: &mut wgpu::RenderPass<'_>,
    draw: SpriteDraw<'_>,
    frame_group: &wgpu::BindGroup,
    args: &IndirectArgsBuffers,
    mode: DrawMode,
    resolved: &ResolvedArgs,
) -> DrawStats {
    let plan = draw.plan;
    let mut stats = DrawStats {
        slots_visited: plan.slot_count(),
        readback_words: resolved.words_read(),
        instances_known_to_cpu: if mode.reads_back() { Some(0) } else { None },
        atlas_pages_bound: u32::try_from(draw.pages.len()).unwrap_or(u32::MAX),
        ..DrawStats::default()
    };
    if plan.slots.is_empty() || draw.pages.is_empty() {
        // No page means no texture to sample, which is a scene with no
        // rasterised text and no decoded image in it rather than an error. The
        // slots are still reported as visited — §5.3's sequence is fixed — and
        // reported as *unavailable*, because there is no such thing as a draw
        // call without a bound texture and pretending they were skipped would
        // say the CPU decided something it did not.
        stats.atlas_pages_bound = 0;
        stats.sprite_slots_unavailable = stats.slots_visited;
        return stats;
    }

    pass.set_pipeline(draw.pipeline);
    pass.set_bind_group(0, frame_group, &[]);
    stats.bind_group_binds += 1;

    let mut known = 0u32;
    for (page_index, page) in draw.pages.iter().enumerate() {
        // `slots_visited` is the length of the *fixed sequence*, which the page
        // loop repeats rather than lengthens — so a skip is counted once, on the
        // first page, and `slots_skipped + (this pass's draws / pages)` still
        // equals it. `atlas_pages_bound` carries the multiplier separately.
        let count_skips = page_index == 0;
        pass.set_bind_group(2, page, &[]);
        stats.bind_group_binds += 1;
        match mode {
            DrawMode::MultiDrawIndirectCount => {
                pass.set_bind_group(1, &plan.bind_group, &[plan.zero_offset]);
                stats.bind_group_binds += 1;
                pass.multi_draw_indirect_count(
                    &args.packed_args,
                    0,
                    &args.draw_count,
                    0,
                    plan.slot_count(),
                );
                stats.draw_calls_issued += 1;
                stats.sprite_draws_issued += 1;
            }
            DrawMode::MultiDrawIndirect => {
                pass.set_bind_group(1, &plan.bind_group, &[plan.zero_offset]);
                stats.bind_group_binds += 1;
                pass.multi_draw_indirect(&args.args, 0, plan.slot_count());
                stats.draw_calls_issued += 1;
                stats.sprite_draws_issued += 1;
            }
            DrawMode::PerSlotIndirect => {
                for index in 0..plan.slots.len() {
                    pass.set_bind_group(1, &plan.bind_group, &[index as u32 * plan.stride]);
                    stats.bind_group_binds += 1;
                    pass.draw_indirect(&args.args, index as u64 * DRAW_INDIRECT_ARGS_STRIDE as u64);
                    stats.draw_calls_issued += 1;
                    stats.sprite_draws_issued += 1;
                }
            }
            DrawMode::CpuReadback => {
                for (index, record) in resolved.records().iter().enumerate() {
                    if record.is_empty() {
                        if count_skips {
                            stats.slots_skipped += 1;
                        }
                        continue;
                    }
                    pass.set_bind_group(1, &plan.bind_group, &[index as u32 * plan.stride]);
                    stats.bind_group_binds += 1;
                    pass.draw(0..record.vertex_count, 0..record.instance_count);
                    stats.draw_calls_issued += 1;
                    stats.sprite_draws_issued += 1;
                    known += record.instance_count;
                }
            }
        }
    }
    if mode.reads_back() {
        stats.instances_known_to_cpu = Some(known);
    }
    stats
}

/// One composite entry, ready to draw: its parameters, its texture, and where
/// its argument record lives.
///
/// The type §5.5's Gap 2 unifies onto. Both producers build one of these and
/// nothing downstream can tell them apart — see
/// `render/textures/external_surface.rs` for the two constructors and the one
/// thing that differs between them.
pub struct PreparedComposite {
    /// The entry's own bind group: parameters, texture view, sampler.
    pub bind_group: wgpu::BindGroup,
    /// Which record in the composite argument buffer belongs to this entry.
    pub slot_index: u32,
}

/// Issue the composite entries' draws, through the same argument buffer and the
/// same fixed sequence the quad pipeline uses.
///
/// `entries` holds only the entries the layer tier kept
/// (`wgpui_core::boundary::compositor::visible_composites`): a culled entry
/// never reaches here, so it costs no bind group, no texture fetch, and — for
/// an external surface — no interaction with `SurfaceRegistry` whatsoever.
/// `plan` carries the prepared entries beside the two counts of entries that
/// did not make it this far, so the stats can account for every entry the frame
/// was given. Taking the plan rather than its three fields separately is what
/// keeps this signature inside clippy's argument limit *and* what stops a caller
/// pairing one frame's entries with another's counts.
pub fn issue_composites(
    pass: &mut wgpu::RenderPass<'_>,
    pipeline: &CompositePipeline,
    frame_group: &wgpu::BindGroup,
    args: &IndirectArgsBuffers,
    plan: &CompositePlan,
    mode: DrawMode,
    resolved: &ResolvedArgs,
) -> DrawStats {
    let entries = &plan.prepared;
    let mut stats = DrawStats {
        composite_entries_visited: u32::try_from(entries.len()).unwrap_or(u32::MAX)
            + plan.culled
            + plan.unavailable,
        composite_entries_culled: plan.culled,
        composite_entries_unavailable: plan.unavailable,
        readback_words: resolved.words_read(),
        instances_known_to_cpu: if mode.reads_back() { Some(0) } else { None },
        ..DrawStats::default()
    };
    if entries.is_empty() {
        return stats;
    }

    pass.set_pipeline(&pipeline.pipeline);
    pass.set_bind_group(0, frame_group, &[]);
    stats.bind_group_binds += 1;

    // Deliberately per entry even where multi-draw is available: each entry
    // binds its own texture, and a bind group cannot change inside a
    // `multi_draw_indirect`. That is a property of compositing textures, not a
    // shortcoming of the mode, and it is why §5.3's O(layer slots) claim is
    // about slots rather than about draw calls in total.
    let mut known = 0u32;
    for entry in entries {
        pass.set_bind_group(1, &entry.bind_group, &[]);
        stats.bind_group_binds += 1;
        let offset = u64::from(entry.slot_index) * DRAW_INDIRECT_ARGS_STRIDE as u64;
        if mode.reads_back() {
            let record = resolved
                .records()
                .get(entry.slot_index as usize)
                .copied()
                .unwrap_or_default();
            if record.is_empty() {
                stats.slots_skipped += 1;
                continue;
            }
            pass.draw(0..record.vertex_count, 0..record.instance_count);
            known += record.instance_count;
        } else {
            pass.draw_indirect(&args.args, offset);
        }
        stats.composite_draws_issued += 1;
        stats.draw_calls_issued += 1;
    }
    if mode.reads_back() {
        stats.instances_known_to_cpu = Some(known);
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_best_available_mode_falls_back_rather_than_failing() {
        assert_eq!(
            DrawMode::best_available(IndirectSupport::NONE),
            DrawMode::PerSlotIndirect,
            "a device with no indirect features still takes the indirect path — \
             the per-slot form needs none"
        );
        assert_eq!(
            DrawMode::best_available(IndirectSupport {
                first_instance: true,
                multi_draw_count: true,
            }),
            DrawMode::MultiDrawIndirectCount
        );
        assert_eq!(
            DrawMode::best_available(IndirectSupport {
                first_instance: true,
                multi_draw_count: false,
            }),
            DrawMode::PerSlotIndirect,
            "without the count feature multi-draw is emulated as a CPU loop, so \
             it is not an improvement over issuing the loop ourselves"
        );
    }

    #[test]
    fn only_the_multi_draw_modes_ask_for_a_nonzero_first_instance() {
        for mode in DrawMode::ALL {
            let expected = match mode {
                DrawMode::MultiDrawIndirect | DrawMode::MultiDrawIndirectCount => {
                    FirstInstance::SlotBase
                }
                _ => FirstInstance::Zero,
            };
            assert_eq!(mode.first_instance(), expected, "{}", mode.name());
            assert_eq!(
                mode.is_available(IndirectSupport::NONE),
                expected == FirstInstance::Zero,
                "{} availability on a featureless device",
                mode.name()
            );
        }
    }

    #[test]
    fn unknown_instance_counts_stay_unknown_when_merged() {
        let mut indirect = DrawStats {
            slots_visited: 4,
            instances_known_to_cpu: None,
            ..DrawStats::default()
        };
        indirect.merge(DrawStats {
            slots_visited: 2,
            instances_known_to_cpu: Some(17),
            ..DrawStats::default()
        });
        assert_eq!(indirect.slots_visited, 6);
        assert_eq!(
            indirect.instances_known_to_cpu, None,
            "a frame that took an indirect path anywhere did not learn its own \
             instance count, and must not report a partial sum as if it had"
        );

        let mut known = DrawStats {
            instances_known_to_cpu: Some(3),
            ..DrawStats::default()
        };
        known.merge(DrawStats {
            instances_known_to_cpu: Some(4),
            ..DrawStats::default()
        });
        assert_eq!(known.instances_known_to_cpu, Some(7));
    }

    #[test]
    fn the_fallback_is_the_only_mode_that_reads_anything_back() {
        assert!(DrawMode::CpuReadback.reads_back());
        assert!(
            DrawMode::ALL
                .iter()
                .filter(|mode| mode.reads_back())
                .count()
                == 1
        );
    }
}
