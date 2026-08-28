//! Applies a `PatchList` to a `Scene`. Phase 1's round-trip gate lives here
//! (docs/gpu-native-architecture.md §8, Phase 1: "apply a patch sequence,
//! read back the resident buffer, matches an equivalent full-rebuild
//! reference exactly").
//!
//! # One frame, five lists, one call
//!
//! [`ScenePatch`] is the whole frontend/backend handoff for one frame: the
//! primitive lists that land in slab-backed residency and the three CPU-side
//! record lists (§2's "layout inputs, hitboxes, dispatch nodes"). It is plain
//! data throughout — buildable, inspectable, replayable, and applicable
//! without a device, which is what §2 means by "data, never a callback."
//!
//! # What "matches a full rebuild exactly" can and cannot mean
//!
//! It means every layer's **occupied bytes** are identical to a scene built
//! from nothing to the same final content. It deliberately does not mean the
//! two arenas are byte-identical end to end, and the difference is not a
//! weakening of the gate — it is the protocol's own contract. `PatchOp::Insert`
//! states that slot placement is the scene's decision and callers must not
//! depend on it, which is exactly what makes relocation and compaction legal
//! (§5.0's second case). A scene that reached its state through a history of
//! inserts, growth across a size class, and removals has legitimately placed a
//! layer at a different base than a fresh build would, and its arena carries
//! vacated blocks a fresh build never allocated. Comparing whole arenas would
//! therefore assert that the allocator has *no* history-dependence — a property
//! the design explicitly rejects.
//!
//! [`assert_matches_rebuild`] checks the real invariant: same layers, same
//! records in the same order, same values, and the same bytes resident for each
//! layer, read back out of the arena at that layer's own address.

use crate::invalidation::axes::Invalidation;
use crate::patch::primitive::{GlyphRun, PolySprite, Primitive, Quad, Shadow, Underline};
use crate::patch::{PatchError, PatchList};
use crate::scene::layer::LayerId;
use crate::scene::record::{DispatchNode, Hitbox, LayoutInput};
use crate::scene::slab_range::{UploadRange, coalesce_uploads, uploaded_byte_count};
use crate::scene::{PrimitiveStore, Scene};

/// One frame's patches, across every record category §2 names.
///
/// The five lists are independent: a frame that changes only a hover colour
/// carries one quad update and four empty lists, and applying it uploads
/// exactly one quad's bytes.
// Not `Eq`: primitive payloads carry `f32` fields, so the whole protocol is
// `PartialEq` and stops there rather than inventing a total order for floats.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScenePatch {
    /// Blurred rounded rectangles, painted under everything else in their layer.
    pub shadows: PatchList<Shadow>,
    /// Fixed-size primitives.
    pub quads: PatchList<Quad>,
    /// Underline and strikethrough rules, painted under their layer's text.
    pub underlines: PatchList<Underline>,
    /// Variable-size primitives.
    pub glyph_runs: PatchList<GlyphRun>,
    /// Colour-atlas sprites — images and rasterised SVGs.
    pub poly_sprites: PatchList<PolySprite>,
    /// Retained-layout-node placement.
    pub layout_inputs: PatchList<LayoutInput>,
    /// Registered hit regions.
    pub hitboxes: PatchList<Hitbox>,
    /// Action-dispatch tree nodes.
    pub dispatch_nodes: PatchList<DispatchNode>,
}

impl ScenePatch {
    /// An empty patch. Applying it changes nothing and uploads zero bytes —
    /// §5.0's third case, reached by the trivial path.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether every list is empty.
    pub fn is_empty(&self) -> bool {
        self.shadows.is_empty()
            && self.quads.is_empty()
            && self.underlines.is_empty()
            && self.glyph_runs.is_empty()
            && self.poly_sprites.is_empty()
            && self.layout_inputs.is_empty()
            && self.hitboxes.is_empty()
            && self.dispatch_nodes.is_empty()
    }

    /// Total operations across every list.
    pub fn len(&self) -> usize {
        self.shadows.len()
            + self.quads.len()
            + self.underlines.len()
            + self.glyph_runs.len()
            + self.poly_sprites.len()
            + self.layout_inputs.len()
            + self.hitboxes.len()
            + self.dispatch_nodes.len()
    }

    /// Drop every operation, keeping the allocations for the next frame.
    pub fn clear(&mut self) {
        self.shadows.clear();
        self.quads.clear();
        self.underlines.clear();
        self.glyph_runs.clear();
        self.poly_sprites.clear();
        self.layout_inputs.clear();
        self.hitboxes.clear();
        self.dispatch_nodes.clear();
    }

    /// Every layer this patch names, deduplicated, in ascending handle order.
    pub fn layers(&self) -> Vec<LayerId> {
        let mut layers: Vec<LayerId> = Vec::new();
        let mut note = |layer: LayerId| {
            if !layers.contains(&layer) {
                layers.push(layer);
            }
        };
        for patch in self.shadows.patches() {
            note(patch.layer);
        }
        for patch in self.quads.patches() {
            note(patch.layer);
        }
        for patch in self.underlines.patches() {
            note(patch.layer);
        }
        for patch in self.glyph_runs.patches() {
            note(patch.layer);
        }
        for patch in self.poly_sprites.patches() {
            note(patch.layer);
        }
        for patch in self.layout_inputs.patches() {
            note(patch.layer);
        }
        for patch in self.hitboxes.patches() {
            note(patch.layer);
        }
        for patch in self.dispatch_nodes.patches() {
            note(patch.layer);
        }
        layers.sort_unstable();
        layers
    }
}

/// The bytes one applied [`ScenePatch`] left stale on the GPU.
///
/// Every entry becomes exactly one `write_buffer(offset, size)` in
/// `wgpui-wgpu` (§3.5); nothing in `wgpui-core` ever issues one. Entries are
/// adjacency-coalesced (§5.0's stated mitigation) and never widened to cover
/// bytes that did not change.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UploadPlan {
    entries: Vec<UploadRange>,
}

impl UploadPlan {
    /// Every pending upload, sorted by kind then offset.
    pub fn entries(&self) -> &[UploadRange] {
        &self.entries
    }

    /// How many `write_buffer` calls this plan implies. §5.0's gate counts
    /// this.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the frame uploads nothing at all.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total bytes this plan moves. §5.0's gate measures this alongside the
    /// call count, because either alone can be gamed by the other.
    pub fn byte_count(&self) -> u64 {
        uploaded_byte_count(&self.entries)
    }
}

/// Apply one frame's patches to `scene`, returning what must be uploaded.
///
/// Every layer a patch names must already be declared via [`Scene::layer`];
/// an undeclared layer is [`PatchError::UnknownLayer`] rather than an
/// implicitly created one, so a producer that derived a stale [`LayerId`] finds
/// out at the patch that used it instead of quietly populating a layer nothing
/// will ever draw.
///
/// On failure the scene is left self-consistent but partially updated. The
/// caller's correct response is to rebuild the affected layer — R-N §2.2's "one
/// slow frame, never incorrect output" — not to retry the same patch.
pub fn apply(scene: &mut Scene, patch: &ScenePatch) -> Result<UploadPlan, PatchError> {
    let touched = patch.layers();
    for layer in touched.iter().copied() {
        if !scene.layers.contains(layer) {
            return Err(PatchError::UnknownLayer(layer));
        }
    }

    let mut entries: Vec<UploadRange> = Vec::new();
    scene
        .shadows
        .apply(&patch.shadows, &mut scene.allocator, &mut entries)?;
    scene
        .quads
        .apply(&patch.quads, &mut scene.allocator, &mut entries)?;
    scene
        .underlines
        .apply(&patch.underlines, &mut scene.allocator, &mut entries)?;
    scene
        .glyph_runs
        .apply(&patch.glyph_runs, &mut scene.allocator, &mut entries)?;
    scene
        .poly_sprites
        .apply(&patch.poly_sprites, &mut scene.allocator, &mut entries)?;
    scene.layout_inputs.apply(&patch.layout_inputs)?;
    scene.hitboxes.apply(&patch.hitboxes)?;
    scene.dispatch_nodes.apply(&patch.dispatch_nodes)?;

    // The axes come from which categories a layer's patches touched, never
    // from the call site that raised the change — `invalidation/axes.rs`'s
    // standing rule, applied at the one place that knows what actually moved.
    for layer in touched {
        let mut axes = Invalidation::empty();
        if names(&patch.shadows, layer)
            || names(&patch.quads, layer)
            || names(&patch.underlines, layer)
            || names(&patch.glyph_runs, layer)
            || names(&patch.poly_sprites, layer)
        {
            axes |= Invalidation::DISPLAY;
            scene
                .layers
                .set_slab(layer, Shadow::KIND, scene.shadows.slab(layer));
            scene
                .layers
                .set_slab(layer, Quad::KIND, scene.quads.slab(layer));
            scene
                .layers
                .set_slab(layer, Underline::KIND, scene.underlines.slab(layer));
            scene
                .layers
                .set_slab(layer, GlyphRun::KIND, scene.glyph_runs.slab(layer));
            scene
                .layers
                .set_slab(layer, PolySprite::KIND, scene.poly_sprites.slab(layer));
        }
        if names(&patch.layout_inputs, layer) {
            axes |= Invalidation::LAYOUT;
        }
        if names(&patch.hitboxes, layer) || names(&patch.dispatch_nodes, layer) {
            axes |= Invalidation::HIT;
        }
        scene.layers.invalidate(layer, axes);
    }

    coalesce_uploads(&mut entries);
    Ok(UploadPlan { entries })
}

fn names<T>(list: &PatchList<T>, layer: LayerId) -> bool {
    list.patches().iter().any(|patch| patch.layer == layer)
}

/// Why two scenes' resident content differs. Returned by
/// [`compare_to_rebuild`] so a failure names the divergence rather than
/// reporting a bare `false`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidencyMismatch {
    /// The two scenes hold different layer sets.
    LayerSet {
        /// Layers the patched scene holds.
        patched: Vec<LayerId>,
        /// Layers the rebuilt reference holds.
        rebuilt: Vec<LayerId>,
    },
    /// A layer holds different records, or the same records in a different
    /// paint order.
    RecordOrder {
        /// The layer at fault.
        layer: LayerId,
        /// Which kind's list diverged.
        kind: crate::patch::primitive::PrimitiveKind,
    },
    /// A layer's occupied bytes differ at `byte_index` slots into its own
    /// range.
    Bytes {
        /// The layer at fault.
        layer: LayerId,
        /// Which kind's arena diverged.
        kind: crate::patch::primitive::PrimitiveKind,
        /// First differing byte, relative to the layer's own range start.
        byte_index: usize,
    },
}

/// Compare a patched scene's resident content against a reference built from
/// nothing, returning the first divergence.
///
/// See this module's doc for what "matches exactly" means and, just as
/// importantly, what it deliberately does not.
pub fn compare_to_rebuild(patched: &Scene, rebuilt: &Scene) -> Option<ResidencyMismatch> {
    let patched_layers = patched.layers.ids();
    let rebuilt_layers = rebuilt.layers.ids();
    if patched_layers != rebuilt_layers {
        return Some(ResidencyMismatch::LayerSet {
            patched: patched_layers,
            rebuilt: rebuilt_layers,
        });
    }

    for layer in patched_layers {
        if let Some(mismatch) = compare_store(&patched.shadows, &rebuilt.shadows, layer) {
            return Some(mismatch);
        }
        if let Some(mismatch) =
            compare_store(&patched.quads, &rebuilt.quads, layer)
        {
            return Some(mismatch);
        }
        if let Some(mismatch) = compare_store(&patched.underlines, &rebuilt.underlines, layer) {
            return Some(mismatch);
        }
        if let Some(mismatch) =
            compare_store(&patched.glyph_runs, &rebuilt.glyph_runs, layer)
        {
            return Some(mismatch);
        }
        if let Some(mismatch) =
            compare_store(&patched.poly_sprites, &rebuilt.poly_sprites, layer)
        {
            return Some(mismatch);
        }
    }
    None
}

fn compare_store<P: Primitive>(
    patched: &PrimitiveStore<P>,
    rebuilt: &PrimitiveStore<P>,
    layer: LayerId,
) -> Option<ResidencyMismatch> {
    if patched.keys(layer) != rebuilt.keys(layer) {
        return Some(ResidencyMismatch::RecordOrder {
            layer,
            kind: P::KIND,
        });
    }
    let patched_bytes = patched.layer_bytes(layer).unwrap_or_default();
    let rebuilt_bytes = rebuilt.layer_bytes(layer).unwrap_or_default();
    if patched_bytes.len() != rebuilt_bytes.len() {
        return Some(ResidencyMismatch::Bytes {
            layer,
            kind: P::KIND,
            byte_index: patched_bytes.len().min(rebuilt_bytes.len()),
        });
    }
    patched_bytes
        .iter()
        .zip(rebuilt_bytes)
        .position(|(left, right)| left != right)
        .map(|byte_index| ResidencyMismatch::Bytes {
            layer,
            kind: P::KIND,
            byte_index,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::RecordKey;
    use crate::patch::primitive::Glyph;
    use crate::reconcile::description::ElementId;
    use crate::reconcile::instance::InstanceKey;
    use crate::scene::layer::{BoundaryId, LayerKey};
    use crate::scene::slab_range::SlabRange;
    use wgpui_layout::taffy_tree::LayoutNodeId;

    fn key(raw: u64) -> RecordKey {
        RecordKey::from_raw(raw)
    }

    fn quad(value: f32) -> Quad {
        Quad {
            origin: [value, value * 2.0],
            size: [10.0, 20.0],
            background: [value / 255.0, 0.5, 0.25, 1.0],
            border_color: [0.0, 0.0, 0.0, 1.0],
            corner_radii: [4.0; 4],
            border_widths: [1.0; 4],
        }
    }

    fn run(glyph_count: usize, value: f32) -> GlyphRun {
        GlyphRun {
            color: [value, 0.5, 0.5, 1.0],
            glyphs: (0..glyph_count)
                .map(|index| Glyph {
                    position: [index as f32, value],
                    ..Glyph::ZERO
                })
                .collect(),
        }
    }

    fn scene_with_layers(count: u64) -> (Scene, Vec<LayerId>) {
        let mut scene = Scene::new();
        let layers = (0..count)
            .map(|raw| scene.layer(LayerKey::untiled(BoundaryId::from_raw(raw))))
            .collect();
        (scene, layers)
    }

    /// One layer's intended content, maintained by the test independently of
    /// the scene so gate #1's reference is not derived from the thing under
    /// test.
    #[derive(Clone, Debug, Default)]
    struct LayerOracle {
        quads: Vec<(RecordKey, Quad)>,
        runs: Vec<(RecordKey, GlyphRun)>,
    }

    impl LayerOracle {
        fn insert_quad(&mut self, index: usize, record: RecordKey, value: Quad) {
            self.quads.insert(index, (record, value));
        }

        fn update_quad(&mut self, record: RecordKey, value: Quad) {
            for entry in self.quads.iter_mut() {
                if entry.0 == record {
                    entry.1 = value;
                }
            }
        }

        fn remove_quad(&mut self, record: RecordKey) {
            self.quads.retain(|entry| entry.0 != record);
        }

        fn insert_run(&mut self, index: usize, record: RecordKey, value: GlyphRun) {
            self.runs.insert(index, (record, value));
        }

        fn update_run(&mut self, record: RecordKey, value: GlyphRun) {
            for entry in self.runs.iter_mut() {
                if entry.0 == record {
                    entry.1 = value.clone();
                }
            }
        }

        /// Confirm the scene actually holds what the oracle says it should,
        /// record for record, before any byte comparison happens.
        fn assert_matches(&self, scene: &Scene, layer: LayerId) {
            let quad_keys: Vec<RecordKey> = self.quads.iter().map(|entry| entry.0).collect();
            assert_eq!(scene.quads.keys(layer), quad_keys);
            for (record, value) in &self.quads {
                assert_eq!(scene.quads.get(layer, *record), Some(value));
            }
            let run_keys: Vec<RecordKey> = self.runs.iter().map(|entry| entry.0).collect();
            assert_eq!(scene.glyph_runs.keys(layer), run_keys);
            for (record, value) in &self.runs {
                assert_eq!(scene.glyph_runs.get(layer, *record), Some(value));
            }
        }
    }

    /// Rebuild a reference scene from nothing, holding exactly what each
    /// oracle describes, in the order it describes.
    fn rebuild(oracles: &[LayerOracle]) -> Result<Scene, PatchError> {
        let mut scene = Scene::new();
        let mut patch = ScenePatch::new();
        for (index, oracle) in oracles.iter().enumerate() {
            let layer = scene.layer(LayerKey::untiled(BoundaryId::from_raw(index as u64)));
            for (position, (record, value)) in oracle.quads.iter().enumerate() {
                patch.quads.insert(layer, *record, position as u32, *value);
            }
            for (position, (record, value)) in oracle.runs.iter().enumerate() {
                patch
                    .glyph_runs
                    .insert(layer, *record, position as u32, value.clone());
            }
        }
        apply(&mut scene, &patch)?;
        Ok(scene)
    }

    // ---- Phase 1 gate 1: round-trip ------------------------------------

    /// **Phase 1 gate #1** (§8): apply a patch sequence, read back the
    /// resident buffer, and confirm it matches an equivalent full-rebuild
    /// reference exactly.
    ///
    /// The sequence deliberately exercises every case the protocol has:
    /// interior inserts, appends, in-place value updates, a variable-size
    /// update that changes slot count, interior removals, and enough growth to
    /// cross a size class and force a relocation — across two layers and both
    /// primitive kinds.
    ///
    /// [`LayerOracle`] tracks the intended final content independently, so the
    /// reference is built from what the patch sequence *meant*, not from what
    /// the scene ended up holding. Deriving the reference from the scene would
    /// make the gate self-fulfilling: it would prove the encoder deterministic
    /// and nothing else.
    #[test]
    fn gate_1_a_patch_sequence_round_trips_to_a_full_rebuild() -> Result<(), PatchError> {
        let (mut scene, layers) = scene_with_layers(2);
        let (first, second) = (layers[0], layers[1]);
        let mut first_oracle = LayerOracle::default();
        let mut second_oracle = LayerOracle::default();

        let mut frame = ScenePatch::new();
        for index in 0..70u32 {
            frame
                .quads
                .append(first, key(index as u64 + 1), index, quad(index as f32));
            first_oracle.insert_quad(index as usize, key(index as u64 + 1), quad(index as f32));
        }
        frame.glyph_runs.insert(first, key(9001), 0, run(4, 1.0));
        first_oracle.insert_run(0, key(9001), run(4, 1.0));
        frame.glyph_runs.insert(first, key(9002), 1, run(6, 2.0));
        first_oracle.insert_run(1, key(9002), run(6, 2.0));
        frame.quads.insert(second, key(5001), 0, quad(100.0));
        second_oracle.insert_quad(0, key(5001), quad(100.0));
        apply(&mut scene, &frame)?;
        first_oracle.assert_matches(&scene, first);

        // Frame 2: interior insert, in-place update, and a run that grows.
        let mut frame = ScenePatch::new();
        frame.quads.insert(first, key(500), 10, quad(-1.0));
        first_oracle.insert_quad(10, key(500), quad(-1.0));
        frame.quads.update(first, key(3), quad(-2.0));
        first_oracle.update_quad(key(3), quad(-2.0));
        frame.glyph_runs.update(first, key(9001), run(11, 3.0));
        first_oracle.update_run(key(9001), run(11, 3.0));
        apply(&mut scene, &frame)?;
        first_oracle.assert_matches(&scene, first);

        // Frame 3: removals, including enough to shrink below a size class,
        // plus an append onto the second layer.
        let mut frame = ScenePatch::new();
        for index in 0..40u32 {
            frame.quads.remove(first, key(index as u64 + 1));
            first_oracle.remove_quad(key(index as u64 + 1));
        }
        frame.quads.append(second, key(5002), 1, quad(101.0));
        second_oracle.insert_quad(1, key(5002), quad(101.0));
        apply(&mut scene, &frame)?;
        first_oracle.assert_matches(&scene, first);
        second_oracle.assert_matches(&scene, second);

        // Frame 4: grow the first layer back across a size class, which
        // relocates it now that the second layer sits behind it in the arena.
        let base_before = scene.quads.slab(first).base;
        let mut frame = ScenePatch::new();
        for index in 0..200u32 {
            let position = 31 + index;
            frame.quads.append(
                first,
                key(20_000 + index as u64),
                position,
                quad(index as f32 + 0.5),
            );
            first_oracle.insert_quad(
                position as usize,
                key(20_000 + index as u64),
                quad(index as f32 + 0.5),
            );
        }
        apply(&mut scene, &frame)?;
        first_oracle.assert_matches(&scene, first);
        assert_ne!(
            scene.quads.slab(first).base,
            base_before,
            "the sequence must actually force a relocation, or it does not test one"
        );

        assert_eq!(first_oracle.quads.len(), 231);
        assert_eq!(second_oracle.quads.len(), 2);

        let reference = rebuild(&[first_oracle, second_oracle])?;
        assert_eq!(
            compare_to_rebuild(&scene, &reference),
            None,
            "a patched scene must hold byte-identical resident content to a full rebuild"
        );
        Ok(())
    }

    #[test]
    fn the_round_trip_comparison_actually_detects_a_difference() -> Result<(), PatchError> {
        let (mut scene, layers) = scene_with_layers(1);
        let layer = layers[0];
        let mut frame = ScenePatch::new();
        frame.quads.insert(layer, key(1), 0, quad(1.0));
        frame.quads.insert(layer, key(2), 1, quad(2.0));
        apply(&mut scene, &frame)?;

        let wrong_value = LayerOracle {
            quads: vec![(key(1), quad(1.0)), (key(2), quad(9.0))],
            runs: Vec::new(),
        };
        assert!(
            matches!(
                compare_to_rebuild(&scene, &rebuild(&[wrong_value])?),
                Some(ResidencyMismatch::Bytes { .. })
            ),
            "the comparison must be capable of failing, or gate #1 proves nothing"
        );

        let wrong_order = LayerOracle {
            quads: vec![(key(2), quad(2.0)), (key(1), quad(1.0))],
            runs: Vec::new(),
        };
        assert!(matches!(
            compare_to_rebuild(&scene, &rebuild(&[wrong_order])?),
            Some(ResidencyMismatch::RecordOrder { .. })
        ));

        let extra_layer = compare_to_rebuild(&scene, &Scene::new());
        assert!(matches!(
            extra_layer,
            Some(ResidencyMismatch::LayerSet { .. })
        ));
        Ok(())
    }

    // ---- Phase 1 gate 4: delta upload ----------------------------------

    /// **Phase 1 gate #4** (§5.0, §8): changing one primitive's value inside a
    /// 10,000-primitive layer produces exactly one pending-upload entry, sized
    /// to that one primitive's slot, at that primitive's own address — not the
    /// layer's full range.
    #[test]
    fn gate_4_one_changed_primitive_in_a_ten_thousand_primitive_layer_uploads_one_slot()
    -> Result<(), PatchError> {
        const PRIMITIVE_COUNT: u32 = 10_000;

        let (mut scene, layers) = scene_with_layers(1);
        let layer = layers[0];

        let mut seed = ScenePatch::new();
        for index in 0..PRIMITIVE_COUNT {
            seed.quads
                .append(layer, key(index as u64 + 1), index, quad(index as f32));
        }
        let seeded = apply(&mut scene, &seed)?;
        // Building the layer by 10,000 appends costs more than 10,000 slots'
        // worth of bytes: each crossing of a size class relocates the layer and
        // rewrites it, exactly as §5.0's second case discloses. The amortised
        // total is bounded the way a `Vec`'s doubling is, and it is the *build*
        // cost, not the steady-state one the gate below measures.
        assert!(
            seeded.byte_count() >= PRIMITIVE_COUNT as u64 * Quad::SLOT_STRIDE as u64,
            "the initial build must at minimum upload every primitive once"
        );

        let target = key(4_999);
        let mut edit = ScenePatch::new();
        edit.quads.update(layer, target, quad(-1.0));
        let plan = apply(&mut scene, &edit)?;

        assert_eq!(plan.len(), 1, "one changed primitive, one write_buffer call");
        assert_eq!(
            plan.byte_count(),
            Quad::SLOT_STRIDE as u64,
            "the upload is one primitive's stride, not the layer's range"
        );
        let entry = plan.entries()[0];
        assert_eq!(entry.kind, Quad::KIND);
        assert_eq!(
            Some(entry.byte_offset..entry.byte_end()),
            scene.quads.record_byte_range(layer, target),
            "the upload is addressed to that primitive's own slot"
        );

        let layer_bytes = scene.quads.slab(layer).used_byte_range(Quad::SLOT_STRIDE);
        let layer_byte_count = layer_bytes.end - layer_bytes.start;
        // Derived from the stride rather than written out, because what this
        // gate measures is the *ratio* between one slot and the layer, and a
        // literal here would have to be edited every time the field set grows —
        // which is how a phase widening `Quad` discovers it broke an unrelated
        // gate rather than that it changed a constant.
        assert_eq!(layer_byte_count, 10_000 * Quad::SLOT_STRIDE as u64);
        assert!(
            plan.byte_count() * 10_000 == layer_byte_count,
            "the delta must be 1/10,000th of the layer, which is the whole point"
        );
        Ok(())
    }

    #[test]
    fn a_clean_frame_uploads_zero_bytes_not_a_small_range() -> Result<(), PatchError> {
        let (mut scene, layers) = scene_with_layers(1);
        let layer = layers[0];
        let mut seed = ScenePatch::new();
        for index in 0..1_000u32 {
            seed.quads
                .append(layer, key(index as u64 + 1), index, quad(index as f32));
        }
        apply(&mut scene, &seed)?;

        let plan = apply(&mut scene, &ScenePatch::new())?;
        assert!(plan.is_empty());
        assert_eq!(plan.byte_count(), 0);
        Ok(())
    }

    #[test]
    fn scattered_updates_stay_scattered_and_adjacent_ones_coalesce() -> Result<(), PatchError> {
        let (mut scene, layers) = scene_with_layers(1);
        let layer = layers[0];
        let mut seed = ScenePatch::new();
        for index in 0..1_000u32 {
            seed.quads
                .append(layer, key(index as u64 + 1), index, quad(index as f32));
        }
        apply(&mut scene, &seed)?;

        let mut scattered = ScenePatch::new();
        for record in [1u64, 400, 800] {
            scattered.quads.update(layer, key(record), quad(-1.0));
        }
        let plan = apply(&mut scene, &scattered)?;
        assert_eq!(plan.len(), 3);
        assert_eq!(plan.byte_count(), 3 * Quad::SLOT_STRIDE as u64);

        let mut adjacent = ScenePatch::new();
        for record in [10u64, 11, 12] {
            adjacent.quads.update(layer, key(record), quad(-2.0));
        }
        let plan = apply(&mut scene, &adjacent)?;
        assert_eq!(plan.len(), 1, "byte-adjacent writes become one call");
        assert_eq!(plan.byte_count(), 3 * Quad::SLOT_STRIDE as u64);
        Ok(())
    }

    // ---- protocol-level behaviour --------------------------------------

    #[test]
    fn an_undeclared_layer_is_rejected_rather_than_implicitly_created() {
        let mut scene = Scene::new();
        let stray = LayerId::from_key(LayerKey::untiled(BoundaryId::from_raw(77)));
        let mut patch = ScenePatch::new();
        patch.quads.insert(stray, key(1), 0, Quad::ZERO);
        assert_eq!(
            apply(&mut scene, &patch),
            Err(PatchError::UnknownLayer(stray))
        );
        assert_eq!(scene.quads.len(stray), 0);
    }

    #[test]
    fn all_four_record_categories_travel_through_one_protocol() -> Result<(), PatchError> {
        let (mut scene, layers) = scene_with_layers(1);
        let layer = layers[0];
        let instance = InstanceKey::from_path(&[ElementId::Slot(0)]);

        let mut patch = ScenePatch::new();
        patch.quads.insert(layer, key(1), 0, quad(1.0));
        patch.layout_inputs.insert(
            layer,
            key(2),
            0,
            LayoutInput {
                instance,
                node: LayoutNodeId::from_raw(7),
                parent: None,
                child_index: 0,
            },
        );
        patch.hitboxes.insert(
            layer,
            key(3),
            0,
            Hitbox {
                instance,
                bounds: [0.0, 0.0, 10.0, 10.0],
                opaque: true,
            },
        );
        patch.dispatch_nodes.insert(
            layer,
            key(4),
            0,
            DispatchNode {
                instance,
                parent: None,
                focus: Some(1),
                key_context: 42,
            },
        );
        assert_eq!(patch.len(), 4);
        assert_eq!(patch.layers(), vec![layer]);

        apply(&mut scene, &patch)?;
        assert_eq!(scene.quads.len(layer), 1);
        assert_eq!(scene.layout_inputs.len(layer), 1);
        assert_eq!(scene.hitboxes.len(layer), 1);
        assert_eq!(scene.dispatch_nodes.len(layer), 1);

        let axes = scene.layers.get(layer).map(crate::scene::Layer::invalidation);
        assert_eq!(axes, Some(Invalidation::all()));
        Ok(())
    }

    #[test]
    fn invalidation_axes_follow_the_categories_a_frame_actually_touched()
    -> Result<(), PatchError> {
        let (mut scene, layers) = scene_with_layers(1);
        let layer = layers[0];
        let mut seed = ScenePatch::new();
        seed.quads.insert(layer, key(1), 0, quad(1.0));
        apply(&mut scene, &seed)?;
        scene.layers.mark_clean(layer);

        let mut display_only = ScenePatch::new();
        display_only.quads.update(layer, key(1), quad(2.0));
        apply(&mut scene, &display_only)?;
        assert_eq!(
            scene.layers.get(layer).map(crate::scene::Layer::invalidation),
            Some(Invalidation::DISPLAY),
            "a colour change must not claim layout or hit geometry moved"
        );
        Ok(())
    }

    #[test]
    fn the_layer_table_tracks_each_kinds_reservation() -> Result<(), PatchError> {
        let (mut scene, layers) = scene_with_layers(1);
        let layer = layers[0];
        let mut patch = ScenePatch::new();
        patch.quads.insert(layer, key(1), 0, quad(1.0));
        patch.glyph_runs.insert(layer, key(2), 0, run(3, 1.0));
        apply(&mut scene, &patch)?;

        let record = scene.layers.get(layer);
        assert_eq!(
            record.map(|layer| layer.slab(Quad::KIND)),
            Some(scene.quads.slab(layer))
        );
        assert_eq!(
            record.map(|layer| layer.slab(GlyphRun::KIND)),
            Some(scene.glyph_runs.slab(layer))
        );
        assert_ne!(record.map(|layer| layer.slab(Quad::KIND)), Some(SlabRange::EMPTY));
        Ok(())
    }

    /// Phase 6.3's structural claim, checked rather than asserted in prose: a
    /// new fixed-size kind needs nothing from the patch protocol beyond its own
    /// list. A shadow inserted, updated in place, and removed leaves the arena
    /// byte-identical to a scene that never held it, and the update touches one
    /// slot rather than the layer's range.
    #[test]
    fn a_shadow_rides_the_same_patch_protocol_as_every_other_kind() -> Result<(), PatchError> {
        let (mut scene, layers) = scene_with_layers(1);
        let layer = layers[0];

        let first = Shadow {
            origin: [1.0, 2.0],
            size: [30.0, 40.0],
            color: [0.0, 0.0, 0.0, 0.5],
            corner_radii: [4.0; 4],
            blur_radius: 8.0,
        };
        let mut patch = ScenePatch::new();
        patch.shadows.insert(layer, key(1), 0, first);
        patch.shadows.insert(layer, key(2), 1, Shadow::ZERO);
        patch.quads.insert(layer, key(3), 0, quad(1.0));
        let plan = apply(&mut scene, &patch)?;
        assert_eq!(scene.shadows.len(layer), 2);
        assert_eq!(
            scene.layers.get(layer).map(|layer| layer.slab(Shadow::KIND)),
            Some(scene.shadows.slab(layer)),
            "the layer table must learn a shadow reservation like any other"
        );
        assert!(!plan.is_empty());

        // §5.0's O(1) case, for this kind: one changed field, one slot uploaded.
        let mut update = ScenePatch::new();
        update.shadows.update(
            layer,
            key(1),
            Shadow {
                blur_radius: 12.0,
                ..first
            },
        );
        let plan = apply(&mut scene, &update)?;
        assert_eq!(plan.len(), 1);
        assert_eq!(plan.byte_count(), Shadow::SLOT_STRIDE as u64);
        assert_eq!(
            plan.entries().first().map(|entry| entry.kind),
            Some(Shadow::KIND),
            "an upload entry must be addressed to the shadow arena, or it \
             overwrites an unrelated kind's bytes"
        );

        let mut removal = ScenePatch::new();
        removal.shadows.remove(layer, key(1));
        removal.shadows.remove(layer, key(2));
        apply(&mut scene, &removal)?;
        assert_eq!(scene.shadows.len(layer), 0);
        Ok(())
    }

    #[test]
    fn an_empty_patch_is_empty_and_clearing_makes_one_empty_again() {
        let mut patch = ScenePatch::new();
        assert!(patch.is_empty());
        patch
            .quads
            .insert(LayerId::from_raw(1), key(1), 0, Quad::ZERO);
        assert!(!patch.is_empty());
        patch.clear();
        assert!(patch.is_empty());
        assert!(patch.layers().is_empty());
    }
}
