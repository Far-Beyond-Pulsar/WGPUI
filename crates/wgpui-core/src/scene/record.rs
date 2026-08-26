//! The non-GPU half of the patch protocol: layout inputs, hitboxes, and
//! dispatch nodes, held in ordered per-layer tables.
//! See docs/gpu-native-architecture.md §2 (the patch list's four categories)
//! and §6 (why these three stay on the CPU on purpose).
//!
//! Not in §3.1's literal file map — a deliberate addition, recorded in
//! `docs/phase-1-results.md`. §2 names four things the patch list carries and
//! §3.1 gives a home to only one of them (`patch/primitive.rs`); the other
//! three need a store, and giving them a shared generic one is what keeps
//! "insert/update/remove for primitives, layout inputs, hitboxes, dispatch
//! nodes" a single protocol instead of four.
//!
//! These records have no slab, no bytes, and no upload: hit-testing, focus,
//! actions, and input dispatch stay on the CPU (§6), and Taffy stays on the
//! CPU for heterogeneous content (§6, §6.1). What they share with primitives
//! is *identity and lifecycle* — the same [`RecordKey`], the same
//! insert/update/remove ops, the same per-layer ordering — which is exactly
//! what a patch protocol is for.

use crate::patch::{Patch, PatchError, PatchList, PatchOp, RecordKey};
use crate::reconcile::instance::InstanceKey;
use crate::scene::layer::LayerId;
use std::collections::HashMap;
use wgpui_layout::taffy_tree::LayoutNodeId;

/// One element's contribution to the layout tree.
///
/// This is the "layout inputs" category of §2's patch list: what a frame tells
/// the persistent Taffy tree about where an element's node sits. The style
/// itself is not carried here — it is applied to the retained node by
/// `wgpui-layout` during reconciliation, and re-sending it would defeat the
/// node reuse §4.0 exists to get.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutInput {
    /// The element that owns this node.
    pub instance: InstanceKey,
    /// The retained Taffy node.
    pub node: LayoutNodeId,
    /// The node's parent, or `None` for the tree root.
    pub parent: Option<LayoutNodeId>,
    /// Position among the parent's children.
    pub child_index: u32,
}

/// A registered hit region (R-N §5.2's point-transform hit test, kept as-is).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hitbox {
    /// The element that registered this region.
    pub instance: InstanceKey,
    /// `[x, y, width, height]` in the owning layer's coordinate space.
    pub bounds: [f32; 4],
    /// Whether this region stops hit-testing from reaching what is behind it.
    pub opaque: bool,
}

/// A node in the action-dispatch tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchNode {
    /// The element that registered this node.
    pub instance: InstanceKey,
    /// The enclosing dispatch node, or `None` at the tree root.
    pub parent: Option<RecordKey>,
    /// The focus handle this node carries, if any.
    pub focus: Option<u64>,
    /// Hash of the node's key context, which is what keymap resolution
    /// matches against. Hashed rather than carried whole because a patch is
    /// data that gets compared, not a context that gets read.
    pub key_context: u64,
}

/// An ordered, keyed table of one record category, partitioned by layer.
///
/// Generic over the payload so all three CPU-side categories share one
/// implementation, and so a later phase adding a fourth (tooltip requests,
/// tab stops) adds a field rather than a mechanism.
#[derive(Debug)]
pub struct RecordStore<T> {
    layers: HashMap<LayerId, LayerRecords<T>>,
}

#[derive(Debug)]
struct LayerRecords<T> {
    order: Vec<RecordKey>,
    records: HashMap<RecordKey, T>,
}

impl<T> Default for LayerRecords<T> {
    fn default() -> Self {
        Self {
            order: Vec::new(),
            records: HashMap::new(),
        }
    }
}

impl<T> Default for RecordStore<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> RecordStore<T> {
    /// An empty store.
    pub fn new() -> Self {
        Self {
            layers: HashMap::new(),
        }
    }

    /// Apply every patch in `list`, in order.
    ///
    /// Stops at the first failure. A partially applied list leaves the store
    /// self-consistent but not in the state the producer intended, so the
    /// caller's correct response is a rebuild of the affected layer — R-N
    /// §2.2's "one slow frame, never incorrect output."
    pub fn apply(&mut self, list: &PatchList<T>) -> Result<(), PatchError> {
        for Patch { layer, op } in list.patches() {
            match op {
                PatchOp::Insert { key, index, value } => {
                    self.insert(*layer, *key, *index, value.clone())?;
                }
                PatchOp::Update { key, value } => self.update(*layer, *key, value.clone())?,
                PatchOp::Remove { key } => self.remove(*layer, *key)?,
            }
        }
        Ok(())
    }

    fn insert(
        &mut self,
        layer: LayerId,
        key: RecordKey,
        index: u32,
        value: T,
    ) -> Result<(), PatchError> {
        let records = self.layers.entry(layer).or_default();
        if records.records.contains_key(&key) {
            return Err(PatchError::DuplicateKey { layer, key });
        }
        let len = records.order.len();
        let position = usize::try_from(index).unwrap_or(usize::MAX);
        if position > len {
            return Err(PatchError::IndexOutOfBounds {
                layer,
                index,
                len: u32::try_from(len).unwrap_or(u32::MAX),
            });
        }
        records.order.insert(position, key);
        records.records.insert(key, value);
        Ok(())
    }

    fn update(&mut self, layer: LayerId, key: RecordKey, value: T) -> Result<(), PatchError> {
        let records = self
            .layers
            .get_mut(&layer)
            .ok_or(PatchError::UnknownKey { layer, key })?;
        let slot = records
            .records
            .get_mut(&key)
            .ok_or(PatchError::UnknownKey { layer, key })?;
        *slot = value;
        Ok(())
    }

    fn remove(&mut self, layer: LayerId, key: RecordKey) -> Result<(), PatchError> {
        let records = self
            .layers
            .get_mut(&layer)
            .ok_or(PatchError::UnknownKey { layer, key })?;
        if records.records.remove(&key).is_none() {
            return Err(PatchError::UnknownKey { layer, key });
        }
        records.order.retain(|existing| *existing != key);
        Ok(())
    }

    /// Drop every record belonging to a layer that is going away.
    pub fn remove_layer(&mut self, layer: LayerId) {
        self.layers.remove(&layer);
    }

    /// A layer's records, in registration order.
    pub fn records(&self, layer: LayerId) -> Vec<T> {
        match self.layers.get(&layer) {
            Some(records) => records
                .order
                .iter()
                .filter_map(|key| records.records.get(key))
                .cloned()
                .collect(),
            None => Vec::new(),
        }
    }

    /// A layer's record keys, in registration order.
    pub fn keys(&self, layer: LayerId) -> Vec<RecordKey> {
        match self.layers.get(&layer) {
            Some(records) => records.order.clone(),
            None => Vec::new(),
        }
    }

    /// One record by address.
    pub fn get(&self, layer: LayerId, key: RecordKey) -> Option<&T> {
        self.layers.get(&layer)?.records.get(&key)
    }

    /// How many records a layer holds.
    pub fn len(&self, layer: LayerId) -> u32 {
        match self.layers.get(&layer) {
            Some(records) => u32::try_from(records.order.len()).unwrap_or(u32::MAX),
            None => 0,
        }
    }

    /// Whether a layer holds no records.
    pub fn is_empty(&self, layer: LayerId) -> bool {
        self.len(layer) == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconcile::description::ElementId;

    fn hitbox(x: f32) -> Hitbox {
        Hitbox {
            instance: InstanceKey::from_path(&[ElementId::Slot(0)]),
            bounds: [x, 0.0, 10.0, 10.0],
            opaque: true,
        }
    }

    fn key(raw: u64) -> RecordKey {
        RecordKey::from_raw(raw)
    }

    #[test]
    fn an_empty_store_reports_nothing_for_any_layer() {
        let store: RecordStore<Hitbox> = RecordStore::new();
        let layer = LayerId::from_raw(1);
        assert!(store.is_empty(layer));
        assert_eq!(store.len(layer), 0);
        assert!(store.records(layer).is_empty());
        assert!(store.get(layer, key(1)).is_none());
    }

    #[test]
    fn records_keep_registration_order_including_interior_inserts() {
        let layer = LayerId::from_raw(1);
        let mut store = RecordStore::new();
        let mut list = PatchList::new();
        list.insert(layer, key(1), 0, hitbox(1.0))
            .insert(layer, key(2), 1, hitbox(2.0))
            .insert(layer, key(3), 1, hitbox(3.0));
        assert_eq!(store.apply(&list), Ok(()));
        assert_eq!(store.keys(layer), vec![key(1), key(3), key(2)]);
    }

    #[test]
    fn removing_an_interior_record_closes_the_gap() {
        let layer = LayerId::from_raw(1);
        let mut store = RecordStore::new();
        let mut list = PatchList::new();
        list.insert(layer, key(1), 0, hitbox(1.0))
            .insert(layer, key(2), 1, hitbox(2.0))
            .insert(layer, key(3), 2, hitbox(3.0))
            .remove(layer, key(2));
        assert_eq!(store.apply(&list), Ok(()));
        assert_eq!(store.keys(layer), vec![key(1), key(3)]);
        assert_eq!(store.len(layer), 2);
    }

    #[test]
    fn a_duplicate_insert_is_rejected_rather_than_silently_overwriting() {
        let layer = LayerId::from_raw(1);
        let mut store = RecordStore::new();
        let mut list = PatchList::new();
        list.insert(layer, key(1), 0, hitbox(1.0))
            .insert(layer, key(1), 1, hitbox(2.0));
        assert_eq!(
            store.apply(&list),
            Err(PatchError::DuplicateKey {
                layer,
                key: key(1)
            })
        );
    }

    #[test]
    fn an_out_of_range_insert_index_is_rejected() {
        let layer = LayerId::from_raw(1);
        let mut store: RecordStore<Hitbox> = RecordStore::new();
        let mut list = PatchList::new();
        list.insert(layer, key(1), 3, hitbox(1.0));
        assert_eq!(
            store.apply(&list),
            Err(PatchError::IndexOutOfBounds {
                layer,
                index: 3,
                len: 0
            })
        );
    }

    #[test]
    fn updating_or_removing_an_unknown_record_is_rejected() {
        let layer = LayerId::from_raw(1);
        let mut store = RecordStore::new();
        let mut seed = PatchList::new();
        seed.insert(layer, key(1), 0, hitbox(1.0));
        assert_eq!(store.apply(&seed), Ok(()));

        let mut update = PatchList::new();
        update.update(layer, key(9), hitbox(9.0));
        assert_eq!(
            store.apply(&update),
            Err(PatchError::UnknownKey {
                layer,
                key: key(9)
            })
        );

        let mut remove = PatchList::new();
        remove.remove(layer, key(9));
        assert_eq!(
            store.apply(&remove),
            Err(PatchError::UnknownKey {
                layer,
                key: key(9)
            })
        );
    }

    #[test]
    fn dropping_a_layer_drops_its_records() {
        let layer = LayerId::from_raw(1);
        let mut store = RecordStore::new();
        let mut list = PatchList::new();
        list.insert(layer, key(1), 0, hitbox(1.0));
        assert_eq!(store.apply(&list), Ok(()));
        store.remove_layer(layer);
        assert!(store.is_empty(layer));
    }
}
