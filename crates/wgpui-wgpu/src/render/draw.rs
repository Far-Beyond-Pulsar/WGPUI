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
//! upload, or a count. [`QuadDrawPlan`] is built from a
//! `wgpui_core::indirect::SlotTable`, which is built from each layer's
//! `SlabRange`, and issuing is a loop over slots whose body is a constant
//! number of calls. [`DrawStats`] then makes it *measurable* rather than
//! asserted, in the style `render_stats` established in the legacy backend.
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
use crate::render::pipelines::{CompositePipeline, QuadPipeline};
use crate::render::readback::{ReadbackError, StagingReader};

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
    /// encoder and blocks. That ordering is the fallback's real cost and it is
    /// visible here rather than buried: an indirect path issues its draws
    /// without ever waiting on the device.
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
pub struct QuadDrawPlan {
    slots: Vec<DrawSlot>,
    bind_group: wgpu::BindGroup,
    stride: u32,
    /// Offset of the entry holding zero, which the multi-draw modes bind
    /// because their records carry the base themselves.
    zero_offset: u32,
}

impl QuadDrawPlan {
    /// Build the plan for one kind's slots.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &QuadPipeline,
        slots: &[DrawSlot],
    ) -> QuadDrawPlan {
        let stride = pipeline.slot_stride;
        // One entry per slot, plus a trailing zero entry for the multi-draw
        // modes.
        let entries = slots.len() + 1;
        let size = u64::from(stride) * entries as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quad slot bases"),
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
        let bind_group = pipeline.slot_bind_group(device, &buffer);
        QuadDrawPlan {
            slots: slots.to_vec(),
            bind_group,
            stride,
            zero_offset: u32::try_from(slots.len()).unwrap_or(0) * stride,
        }
    }

    /// Entries in the fixed sequence.
    pub fn slot_count(&self) -> u32 {
        u32::try_from(self.slots.len()).unwrap_or(u32::MAX)
    }

    /// The slots this plan draws.
    pub fn slots(&self) -> &[DrawSlot] {
        &self.slots
    }
}

/// Issue one kind's fixed draw sequence.
///
/// `frame_group` is [`QuadPipeline::frame_bind_group`]'s output for this
/// frame's globals, arena, and indirection buffer; `args` holds the records the
/// compute pass wrote. Nothing else is read, and in particular nothing about
/// the scene's contents is.
pub fn issue_quads(
    pass: &mut wgpu::RenderPass<'_>,
    pipeline: &QuadPipeline,
    plan: &QuadDrawPlan,
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

    pass.set_pipeline(&pipeline.pipeline);
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
/// `culled` and `unavailable` are how many entries did not make it this far,
/// carried in only so the stats can account for every entry the frame was
/// given.
pub fn issue_composites(
    pass: &mut wgpu::RenderPass<'_>,
    pipeline: &CompositePipeline,
    frame_group: &wgpu::BindGroup,
    args: &IndirectArgsBuffers,
    entries: &[PreparedComposite],
    culled: u32,
    unavailable: u32,
    mode: DrawMode,
    resolved: &ResolvedArgs,
) -> DrawStats {
    let mut stats = DrawStats {
        composite_entries_visited: u32::try_from(entries.len()).unwrap_or(u32::MAX)
            + culled
            + unavailable,
        composite_entries_culled: culled,
        composite_entries_unavailable: unavailable,
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
