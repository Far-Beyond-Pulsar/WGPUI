//! Patch-list protocol: `Patch`, `PatchList` — the one frontend/backend
//! boundary (docs/gpu-native-architecture.md §2, §5.0).
//!
//! The load-bearing property, quoted from §2: the patch list is **data, never
//! a callback or a control-flow handoff.** Nothing in this module borrows a
//! scene, calls back into the producer, or runs user code. A `PatchList` can
//! be built on one side of the boundary, inspected, serialised, replayed, and
//! applied on the other — which is exactly what makes the backend swappable
//! while the frontend that produces it stays untouched (§7).
//!
//! # One protocol, four record categories
//!
//! Phase 1's scope names four things the patch list carries: primitives,
//! layout inputs, hitboxes, and dispatch nodes. Rather than four hand-written
//! protocols, [`PatchOp`] is generic over its payload and all four use it. The
//! difference between them is not the *protocol* but the *store* that consumes
//! it: primitives land in a slab-backed
//! [`crate::scene::PrimitiveStore`] that encodes them to bytes and reports
//! upload ranges (§5.0), while the other three land in a plain keyed
//! [`crate::scene::RecordStore`] with no GPU residency at all.
//!
//! # Addressing
//!
//! Every record is addressed by a [`RecordKey`] that is stable across frames,
//! derived from the reconciler's [`crate::reconcile::instance::InstanceKey`]
//! plus an ordinal for elements that emit more than one record of a kind. This
//! is §5.0's "naming a GPU-side address per primitive, not just a per-layer
//! range" — the extension that makes an O(1) delta upload expressible at all.

pub mod apply;
pub mod emit;
pub mod primitive;

use crate::patch::primitive::{EncodeError, PrimitiveKind};
use crate::reconcile::instance::InstanceKey;
use crate::scene::layer::LayerId;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// The stable, cross-frame address of one patchable record within its layer.
///
/// Derived from the emitting element's [`InstanceKey`] and an ordinal
/// distinguishing several records emitted by the same element (a div emitting
/// a background quad and a border quad, say). Because an `InstanceKey` is
/// itself derived from the element's path — positionally, with no `.id()`
/// required (§4.0) — a record keeps its address across frames without any
/// element in the tree opting into anything.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RecordKey(u64);

impl RecordKey {
    /// Derive the key for the `ordinal`-th record emitted by `instance`.
    pub fn new(instance: InstanceKey, ordinal: u32) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        instance.as_raw().hash(&mut hasher);
        ordinal.hash(&mut hasher);
        // Reserve 0 so a defaulted or zeroed key can never alias a live
        // record, matching the legacy backend's `InstanceKey`/`LayerKey`
        // convention.
        RecordKey(hasher.finish() | 1)
    }

    /// Wrap a raw value. Intended for tests and for producers that already
    /// have a stable integer identity of their own.
    pub const fn from_raw(raw: u64) -> Self {
        RecordKey(raw)
    }

    /// The raw value.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// One record-level operation. Generic over the payload so primitives, layout
/// inputs, hitboxes, and dispatch nodes share one protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PatchOp<T> {
    /// Add a record at `index` in its layer's ordered list.
    ///
    /// `index` is the position in the layer's paint/registration order, not a
    /// slab slot: slot placement is the scene's decision, and callers must not
    /// depend on it (that is what makes relocation and compaction legal).
    Insert {
        /// Stable address of the new record.
        key: RecordKey,
        /// Position in the layer's ordered list. Must be `<= len`.
        index: u32,
        /// The record's initial value.
        value: T,
    },
    /// Replace a record's value, keeping its position.
    ///
    /// This is §5.0's O(1) case for primitives: a value update that does not
    /// change the record's slot count re-encodes in place and uploads exactly
    /// that record's own bytes.
    Update {
        /// Address of the record to replace.
        key: RecordKey,
        /// The record's new value.
        value: T,
    },
    /// Drop a record from its layer.
    Remove {
        /// Address of the record to drop.
        key: RecordKey,
    },
}

impl<T> PatchOp<T> {
    /// The record this operation addresses.
    pub const fn key(&self) -> RecordKey {
        match self {
            PatchOp::Insert { key, .. } | PatchOp::Update { key, .. } | PatchOp::Remove { key } => {
                *key
            }
        }
    }
}

/// One operation, scoped to the layer it applies to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Patch<T> {
    /// The layer whose ordered record list this operation edits.
    pub layer: LayerId,
    /// What to do.
    pub op: PatchOp<T>,
}

/// An ordered sequence of [`Patch`]es for one record category.
///
/// Order is significant: operations apply in sequence, so an insert followed
/// by an update of the same key is legal and means what it reads as.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatchList<T> {
    patches: Vec<Patch<T>>,
}

impl<T> PatchList<T> {
    /// An empty patch list. A frame in which nothing changed produces one of
    /// these per category, and applying it uploads zero bytes (§5.0's third
    /// case: "a clean layer uploads zero bytes — not a small range, zero").
    pub const fn new() -> Self {
        Self {
            patches: Vec::new(),
        }
    }

    /// Append an insert.
    pub fn insert(&mut self, layer: LayerId, key: RecordKey, index: u32, value: T) -> &mut Self {
        self.patches.push(Patch {
            layer,
            op: PatchOp::Insert { key, index, value },
        });
        self
    }

    /// Append an insert at the end of the layer's current list.
    ///
    /// `len` is the caller's own count of records already in that layer; the
    /// scene validates it, so an out-of-date count is reported as
    /// [`PatchError::IndexOutOfBounds`] rather than silently landing
    /// somewhere else.
    pub fn append(&mut self, layer: LayerId, key: RecordKey, len: u32, value: T) -> &mut Self {
        self.insert(layer, key, len, value)
    }

    /// Append a value update.
    pub fn update(&mut self, layer: LayerId, key: RecordKey, value: T) -> &mut Self {
        self.patches.push(Patch {
            layer,
            op: PatchOp::Update { key, value },
        });
        self
    }

    /// Append a removal.
    pub fn remove(&mut self, layer: LayerId, key: RecordKey) -> &mut Self {
        self.patches.push(Patch {
            layer,
            op: PatchOp::Remove { key },
        });
        self
    }

    /// The operations, in application order.
    pub fn patches(&self) -> &[Patch<T>] {
        &self.patches
    }

    /// How many operations this list carries.
    pub fn len(&self) -> usize {
        self.patches.len()
    }

    /// Whether this list carries no operations.
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }

    /// Drop every operation, keeping the allocation for the next frame.
    pub fn clear(&mut self) {
        self.patches.clear();
    }
}

impl<T> Default for PatchList<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A patch could not be applied. Every variant names the record or layer at
/// fault so a caller can report it rather than guess.
///
/// No variant is recoverable *in place* — a scene that rejects a patch has not
/// applied it — but every variant is recoverable *by rebuild*, which is R-N
/// §2.2's discipline: "a mismatch causes a subtree rebuild — one slow frame,
/// never incorrect output."
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PatchError {
    /// The patch named a layer the scene does not have.
    UnknownLayer(LayerId),
    /// An insert used a key the layer already holds.
    DuplicateKey {
        /// The layer.
        layer: LayerId,
        /// The colliding key.
        key: RecordKey,
    },
    /// An update or remove named a key the layer does not hold.
    UnknownKey {
        /// The layer.
        layer: LayerId,
        /// The missing key.
        key: RecordKey,
    },
    /// An insert index was past the end of the layer's list.
    IndexOutOfBounds {
        /// The layer.
        layer: LayerId,
        /// The requested index.
        index: u32,
        /// The layer's record count at the time.
        len: u32,
    },
    /// The kind's slab arena cannot address the requested slot count.
    ///
    /// Requires a single layer claiming over two billion slots of one kind,
    /// which no viable GPU buffer supports; the caller should surface this
    /// rather than retry.
    SlabOverflow {
        /// The kind whose arena overflowed.
        kind: PrimitiveKind,
        /// Slots the layer asked for.
        requested_slots: u64,
    },
    /// A primitive's `encode` disagreed with its own `SLOT_STRIDE`.
    Encode {
        /// The kind whose encoding failed.
        kind: PrimitiveKind,
        /// What went wrong.
        error: EncodeError,
    },
}

impl std::fmt::Display for PatchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatchError::UnknownLayer(layer) => {
                write!(formatter, "patch names unknown layer {layer:?}")
            }
            PatchError::DuplicateKey { layer, key } => write!(
                formatter,
                "insert of {key:?} into {layer:?} collides with an existing record"
            ),
            PatchError::UnknownKey { layer, key } => {
                write!(formatter, "{layer:?} holds no record {key:?}")
            }
            PatchError::IndexOutOfBounds { layer, index, len } => write!(
                formatter,
                "insert index {index} is past the end of {layer:?} ({len} records)"
            ),
            PatchError::SlabOverflow {
                kind,
                requested_slots,
            } => write!(
                formatter,
                "{kind:?} slab cannot address {requested_slots} slots"
            ),
            PatchError::Encode { kind, error } => write!(formatter, "{kind:?}: {error}"),
        }
    }
}

impl std::error::Error for PatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PatchError::Encode { error, .. } => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::primitive::Quad;
    use crate::reconcile::description::ElementId;

    fn instance(path: &[ElementId]) -> InstanceKey {
        InstanceKey::from_path(path)
    }

    #[test]
    fn record_keys_are_stable_across_frames_and_distinct_per_ordinal() {
        let element = instance(&[ElementId::Slot(0), ElementId::Slot(3)]);
        assert_eq!(RecordKey::new(element, 0), RecordKey::new(element, 0));
        assert_ne!(RecordKey::new(element, 0), RecordKey::new(element, 1));
    }

    #[test]
    fn record_keys_of_different_elements_do_not_collide() {
        let first = instance(&[ElementId::Slot(0)]);
        let second = instance(&[ElementId::Slot(1)]);
        assert_ne!(RecordKey::new(first, 0), RecordKey::new(second, 0));
    }

    #[test]
    fn record_key_is_never_zero() {
        for ordinal in 0..64 {
            assert_ne!(RecordKey::new(instance(&[]), ordinal).as_raw(), 0);
        }
    }

    #[test]
    fn an_empty_patch_list_carries_nothing() {
        let list: PatchList<Quad> = PatchList::new();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        assert!(list.patches().is_empty());
    }

    #[test]
    fn patch_list_preserves_operation_order() {
        let layer = LayerId::from_raw(1);
        let key = RecordKey::from_raw(9);
        let mut list = PatchList::new();
        list.insert(layer, key, 0, Quad::ZERO)
            .update(layer, key, Quad::ZERO)
            .remove(layer, key);
        let kinds: Vec<&'static str> = list
            .patches()
            .iter()
            .map(|patch| match patch.op {
                PatchOp::Insert { .. } => "insert",
                PatchOp::Update { .. } => "update",
                PatchOp::Remove { .. } => "remove",
            })
            .collect();
        assert_eq!(kinds, vec!["insert", "update", "remove"]);
        assert!(list.patches().iter().all(|patch| patch.op.key() == key));
    }
}
