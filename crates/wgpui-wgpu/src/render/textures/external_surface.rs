//! Unified `WgpuSurface` consumer entry (§5.5, Gap 2) — the *consuming*
//! half of `SurfaceRegistry`'s composite path, folded into the same
//! indirect-draw entry mechanism `.boundary()`'s texture-retained layers
//! use. `SurfaceRegistry`'s producer side is untouched (§9's risk table).
//! See docs/gpu-native-architecture.md §5.5, §9.
//!
//! # What Gap 2 actually is, and what closing it costs
//!
//! §5.5's complaint is precise: "Two composite pipelines exist today for one
//! operation." In the legacy renderer both live in the *same* `match` arm on
//! `PrimitiveBatch::Surfaces` and share the `surfaces` pipeline, but they are
//! two separate ~180-line branches — `SurfaceContent::Wgpu(surface_id)` fetches
//! through `SurfaceRegistry`, `SurfaceContent::Layer(layer_id)` fetches through
//! `self.layer_textures`, and each then builds its own params buffer, its own
//! bind group, and issues its own `pass.draw(0..4, 0..1)`.
//!
//! Here they are one function, [`plan_composites`], and the difference between
//! the two producers survives in exactly one expression:
//! [`CompositeConsumer::view`]'s `match`. Everything after it — the parameter
//! block, the bind group, the argument record, the draw call — is common code
//! that cannot tell them apart.
//!
//! # What is deliberately *not* touched
//!
//! §9's risk table names this phase's specific failure mode, so it is worth
//! stating what this file does and does not call. It calls exactly two
//! `SurfaceRegistry` methods, in the order the legacy surfaces batch calls
//! them:
//!
//! 1. [`SurfaceRegistry::swap_ready_display_if_new`] — promote a newly produced
//!    frame, and *only* a newly produced one. The legacy call site's comment
//!    explains why the gate matters (an unconditional swap strobes the canvas
//!    whenever the producer skips a frame) and that reasoning is unchanged.
//! 2. [`SurfaceRegistry::front_view`] — the view to sample.
//!
//! It calls nothing else. Not `swap_rendering_ready`, not `present_synced`, not
//! `resize`, not `set_redraw_pending`, not `has_unconsumed_frame`. The producer
//! paces itself exactly as it did.
//!
//! # The one behavioural change, and it is the point
//!
//! A composite entry the layer tier culled
//! (`wgpui_core::boundary::compositor::visible_composites`) never reaches
//! either call. §5.5 promises that "a 3D viewport fully covered by a modal
//! stops being drawn at all, which it cannot today"; the observable form of
//! that promise is that a covered surface's `frame_generation` stops being
//! consumed, its view is never fetched, and no bind group is built for it.
//! That is a change in what the *consumer* does on a frame where the surface is
//! invisible, and it is what Gap 2 was for.

use wgpui_core::boundary::compositor::{CompositeEntry, CompositeSource, visible_composites};
use wgpui_core::indirect::{DrawSlot, SlotTable};
use wgpui_core::patch::primitive::PrimitiveKind;
use wgpui_core::scene::layer::LayerId;

use crate::render::draw::PreparedComposite;
use crate::render::pipelines::CompositePipeline;
use crate::render::surface_registry::{SurfaceId, SurfaceRegistry};
use crate::render::textures::layer_texture::LayerTexturePool;

/// Where a composite entry's texture is fetched from.
///
/// The one place §5.5's two producers are still distinguishable. Both fields
/// are optional so a frame with no external surfaces need not hold a registry,
/// and a frame with no texture-retained boundaries need not hold a pool.
pub struct CompositeConsumer<'a> {
    /// The externally-owned triple buffers. Untouched by this phase except for
    /// the two consumer-side calls this module's doc lists.
    pub registry: Option<&'a SurfaceRegistry>,
    /// The framework's own baked boundary textures.
    pub textures: Option<&'a LayerTexturePool>,
}

impl CompositeConsumer<'_> {
    /// The view to sample for `source`, or `None` when its producer has nothing
    /// ready.
    ///
    /// For an external surface this is where the compositor promotes a newly
    /// produced frame, which is why it takes `&self` and has an effect: the
    /// promotion is a consumer-side act on producer-side state, and putting it
    /// anywhere else would mean the swap happened for entries that were never
    /// drawn.
    pub fn view(&self, source: CompositeSource) -> Option<wgpu::TextureView> {
        match source {
            CompositeSource::External(id) => {
                let registry = self.registry?;
                let id = SurfaceId::from_raw(id.as_raw());
                // Gated, exactly as the legacy surfaces batch gates it: swap
                // only if the external renderer produced a new frame since the
                // last composite, or `display` rotates to a stale buffer
                // whenever the producer skipped a frame and the canvas strobes.
                registry.swap_ready_display_if_new(id);
                registry.clear_redraw_pending(id);
                registry.front_view(id)
            }
            CompositeSource::BoundaryTexture(boundary) => self.textures?.view(boundary).cloned(),
        }
    }
}

/// One frame's composite entries, resolved into draws.
pub struct CompositePlan {
    /// The entries that survived the layer tier and found a texture, in draw
    /// order, each carrying its own bind group and argument-record index.
    pub prepared: Vec<PreparedComposite>,
    /// Entries the layer tier dropped — covered by opaque content above them.
    pub culled: u32,
    /// Entries whose producer had nothing ready. Not an error: an external
    /// surface that has never presented a frame, or a boundary swept out of the
    /// texture pool, both land here and are simply not drawn this frame.
    pub unavailable: u32,
    /// One word per entry, `1` where it must not draw — the `culled` stream
    /// `IndirectArgsPass` reads, in exactly the shape it reads for primitives.
    pub culled_mask: Vec<u32>,
    /// One layer-local index per entry, all zero: a composite slot holds
    /// exactly one instance, so its draw permutation is the identity.
    pub draw_order: Vec<u32>,
    /// The parameter blocks, kept alive as long as the bind groups that name
    /// them.
    params: Vec<wgpu::Buffer>,
}

impl CompositePlan {
    /// The fixed slot sequence for these entries: one slot each, one instance
    /// each.
    ///
    /// A composite entry is the degenerate case of a (layer, kind) slot — one
    /// reservation of one slot — which is what lets it take the same argument
    /// buffer, the same compute pass, and the same draw sequence as a layer
    /// full of quads.
    pub fn slots(entry_count: usize) -> Vec<DrawSlot> {
        (0..entry_count)
            .map(|index| DrawSlot {
                // The slot's identity is its position; composite entries are not
                // scene layers and have no `LayerId` of their own.
                layer: LayerId::from_raw(index as u64 + 1),
                kind: PrimitiveKind::Quad,
                base: u32::try_from(index).unwrap_or(0),
                count: 1,
            })
            .collect()
    }

    /// The same slots as a [`SlotTable`].
    pub fn slot_table(entry_count: usize) -> SlotTable {
        SlotTable::from_grouped(Self::slots(entry_count)).unwrap_or_default()
    }

    /// How many parameter buffers this plan owns, for a test that wants to know
    /// a culled entry cost nothing.
    pub fn params_allocated(&self) -> usize {
        self.params.len()
    }
}

/// Resolve one frame's composite entries into draws.
///
/// Runs the layer tier first and fetches nothing for an entry it drops — see
/// this module's doc for why that ordering is the whole of §5.5's promised win.
/// `entries` is in draw order.
pub fn plan_composites(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &CompositePipeline,
    consumer: &CompositeConsumer<'_>,
    entries: &[CompositeEntry],
) -> CompositePlan {
    let visible = visible_composites(entries);
    let mut plan = CompositePlan {
        prepared: Vec::new(),
        culled: 0,
        unavailable: 0,
        culled_mask: vec![1; entries.len()],
        draw_order: vec![0; entries.len()],
        params: Vec::new(),
    };

    for (index, entry) in entries.iter().enumerate() {
        if !visible.get(index).copied().unwrap_or(true) {
            plan.culled += 1;
            continue;
        }
        let Some(view) = consumer.view(entry.source) else {
            plan.unavailable += 1;
            continue;
        };

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite entry params"),
            size: CompositePipeline::PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params, 0, &encode_params(entry));
        let bind_group = pipeline.entry_bind_group(device, &params, &view);
        plan.params.push(params);
        plan.prepared.push(PreparedComposite {
            bind_group,
            slot_index: u32::try_from(index).unwrap_or(0),
        });
        if let Some(flag) = plan.culled_mask.get_mut(index) {
            *flag = 0;
        }
    }
    plan
}

/// The 48 bytes `surfaces.wgsl`'s `CompositeParams` expects.
fn encode_params(entry: &CompositeEntry) -> [u8; 48] {
    let mut bytes = [0u8; 48];
    let mask = entry.content_mask;
    let values = [
        entry.bounds.min_x,
        entry.bounds.min_y,
        entry.bounds.width(),
        entry.bounds.height(),
        mask.min_x,
        mask.min_y,
        mask.width(),
        mask.height(),
        entry.opacity,
        entry.corner_radius,
        0.0,
        0.0,
    ];
    for (index, value) in values.iter().enumerate() {
        let offset = index * 4;
        if let Some(slot) = bytes.get_mut(offset..offset + 4) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpui_core::boundary::compositor::ExternalSurfaceId;
    use wgpui_core::geometry::Rect;
    use wgpui_core::scene::layer::BoundaryId;

    use crate::render::device::context_or_report;

    fn window() -> Rect {
        Rect::from_origin_size([0.0, 0.0], [1000.0, 800.0])
    }

    #[test]
    fn a_composite_entry_is_one_slot_holding_one_instance() {
        let slots = CompositePlan::slots(3);
        assert_eq!(slots.len(), 3);
        assert!(slots.iter().all(|slot| slot.count == 1));
        assert_eq!(
            slots.iter().map(|slot| slot.base).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(CompositePlan::slot_table(3).len(), 3);
    }

    #[test]
    fn the_parameter_block_carries_the_entrys_geometry_verbatim() {
        let entry = CompositeEntry {
            opacity: 0.5,
            corner_radius: 8.0,
            ..CompositeEntry::sampled(
                CompositeSource::External(ExternalSurfaceId::from_raw(1)),
                Rect::from_origin_size([10.0, 20.0], [300.0, 200.0]),
                window(),
            )
        };
        let bytes = encode_params(&entry);
        let read = |index: usize| -> f32 {
            let mut word = [0u8; 4];
            word.copy_from_slice(&bytes[index * 4..index * 4 + 4]);
            f32::from_le_bytes(word)
        };
        assert_eq!(
            [read(0), read(1), read(2), read(3)],
            [10.0, 20.0, 300.0, 200.0]
        );
        assert_eq!(
            [read(4), read(5), read(6), read(7)],
            [0.0, 0.0, 1000.0, 800.0]
        );
        assert_eq!(read(8), 0.5);
        assert_eq!(read(9), 8.0);
    }

    /// §5.5's promise, at the level this module owns it: a culled entry costs
    /// no texture fetch, no parameter buffer, and no bind group. The registry
    /// half is asserted in `tests/surface_registry_consumer.rs`, which can
    /// observe that the producer's generation was never consumed.
    #[test]
    fn a_culled_entry_allocates_nothing() {
        let Some(context) = context_or_report("composite_plan_culled") else {
            return;
        };
        let pipeline = CompositePipeline::new(&context.device);
        let mut pool = LayerTexturePool::default();
        pool.begin_frame();
        let covered = BoundaryId::from_raw(1);
        let cover = BoundaryId::from_raw(2);
        pool.acquire(&context.device, covered, 400, 300, 1);
        pool.acquire(&context.device, cover, 1000, 800, 1);

        let entries = [
            CompositeEntry::sampled(
                CompositeSource::BoundaryTexture(covered),
                Rect::from_origin_size([200.0, 150.0], [400.0, 300.0]),
                window(),
            ),
            CompositeEntry {
                source_is_opaque: true,
                ..CompositeEntry::sampled(
                    CompositeSource::BoundaryTexture(cover),
                    window(),
                    window(),
                )
            },
        ];
        let consumer = CompositeConsumer {
            registry: None,
            textures: Some(&pool),
        };
        let plan = plan_composites(
            &context.device,
            &context.queue,
            &pipeline,
            &consumer,
            &entries,
        );
        assert_eq!(plan.culled, 1);
        assert_eq!(plan.prepared.len(), 1);
        assert_eq!(
            plan.params_allocated(),
            1,
            "the culled entry must not have built a parameter buffer"
        );
        assert_eq!(
            plan.culled_mask,
            vec![1, 0],
            "the mask the compute pass reads must agree with what was prepared"
        );
        assert_eq!(
            plan.prepared.first().map(|entry| entry.slot_index),
            Some(1),
            "a prepared entry keeps its position in the fixed sequence rather \
             than being renumbered"
        );
    }

    #[test]
    fn a_producer_with_nothing_ready_is_not_an_error() {
        let Some(context) = context_or_report("composite_plan_unavailable") else {
            return;
        };
        let pipeline = CompositePipeline::new(&context.device);
        let entries = [CompositeEntry::sampled(
            CompositeSource::BoundaryTexture(BoundaryId::from_raw(9)),
            window(),
            window(),
        )];
        let plan = plan_composites(
            &context.device,
            &context.queue,
            &pipeline,
            &CompositeConsumer {
                registry: None,
                textures: None,
            },
            &entries,
        );
        assert_eq!(plan.unavailable, 1);
        assert_eq!(plan.culled, 0);
        assert!(plan.prepared.is_empty());
        assert_eq!(plan.culled_mask, vec![1]);
    }
}
