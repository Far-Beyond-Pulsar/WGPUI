//! Positional-identity fallback (SFD §1.0), reused by `WgpuSurface`
//! (docs/gpu-native-architecture.md §5.5, Gap 1).
//!
//! # The gap this closes, in SFD's own terms
//!
//! SFD §1.0 found layer creation was "100% manual, and not for a stylistic
//! reason": a `LayerKey` was derived from a `GlobalElementId`, and a
//! `GlobalElementId` only existed for an element whose `.id()` returned `Some`.
//! A bare `div()` had no identity at any point in the stack, so `.layer()`
//! without `.id()` compiled, ran, and did nothing (SFD §0.2). The fix SFD
//! prescribes is to extend the positional-fallback pattern the framework
//! already applied to a layer's *children* one level up, to the layer root
//! itself.
//!
//! `wgpui-core` gets that for free on the reconciliation side — the walk in
//! `reconcile/reconciler.rs` already addresses every element by
//! `ElementId::Slot(index)` when it names nothing, so an `InstanceKey` exists
//! for every element unconditionally (§4.0). What Phase 2 adds is the other
//! half: a **boundary** root derives its [`BoundaryId`] from that same path, so
//! declaring `.boundary()` on an anonymous element yields a stable, cross-frame
//! compositing identity with nothing to remember. §4.1's guarantee — "a
//! forgotten `.id()` only costs a boundary its independent-compositing benefit"
//! — becomes, here, a stronger statement still: it costs nothing at all unless
//! the element also moves between sibling slots.
//!
//! # One derivation, two consumers
//!
//! [`BoundaryIdentity::from_path`] hashes the same `&[ElementId]` slice
//! `InstanceKey::from_path` hashes, so a boundary and the element that declared
//! it are addressed from one source of truth. §5.5's Gap 1 names `WgpuSurface`
//! as the second consumer of exactly this mechanism; that element's shape is
//! `wgpui-widgets`' (`wgpu_surface.rs`), and it needs no new identity path
//! because this one already covers it.

use crate::reconcile::description::ElementId;
use crate::scene::layer::{BoundaryId, LayerId, LayerKey};
use std::hash::{Hash, Hasher};

/// Stable identity of a retained scrolling root.
///
/// A scrolling root is distinct from its compositing boundary. Keeping the
/// identity separate lets an inspector describe an untiled root, a tiled
/// root, and a root that currently happens to share a layer without making
/// those implementation details part of the query contract.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScrollRootId(u64);

impl ScrollRootId {
    /// The window's scrolling root.
    pub const ROOT: ScrollRootId = ScrollRootId(0);

    /// Wrap a raw retained identity.
    pub const fn from_raw(raw: u64) -> Self {
        ScrollRootId(raw)
    }

    /// Return the raw retained identity.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// Derives a compositing boundary's cross-frame identity from where it sits.
pub struct BoundaryIdentity;

impl BoundaryIdentity {
    /// The boundary declared by the element at `path`.
    ///
    /// An empty path is the window root itself, which is
    /// [`BoundaryId::ROOT`] — the boundary every element that declared none
    /// belongs to.
    pub fn from_path(path: &[ElementId]) -> BoundaryId {
        if path.is_empty() {
            return BoundaryId::ROOT;
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        // Domain-separated from `InstanceKey`/`StateScope`, which hash the same
        // slice: three identities derived from one path must not collide into
        // each other's tables if any of them is ever used as a raw integer.
        "boundary".hash(&mut hasher);
        path.hash(&mut hasher);
        // Reserve 0 so a derived identity can never alias `BoundaryId::ROOT`,
        // which is not derived from any path.
        BoundaryId::from_raw(hasher.finish() | 1)
    }

    /// The untiled layer holding the boundary declared at `path`.
    ///
    /// Phase 4.5 is what makes a boundary hold more than one layer (§4.3);
    /// until then a boundary and its layer are one to one, and going through
    /// [`LayerKey`] rather than around it is what keeps that true by
    /// construction.
    pub fn layer_for_path(path: &[ElementId]) -> LayerId {
        LayerId::from_key(LayerKey::untiled(Self::from_path(path)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconcile::instance::InstanceKey;

    #[test]
    fn an_anonymous_boundary_still_gets_a_stable_identity() {
        let path = [ElementId::Slot(0), ElementId::Slot(2)];
        assert_eq!(
            BoundaryIdentity::from_path(&path),
            BoundaryIdentity::from_path(&path),
            "SFD §1.0: positional identity is identity, not a degraded substitute"
        );
        assert_ne!(BoundaryIdentity::from_path(&path), BoundaryId::ROOT);
    }

    #[test]
    fn sibling_slots_are_different_boundaries() {
        let first = BoundaryIdentity::from_path(&[ElementId::Slot(0), ElementId::Slot(0)]);
        let second = BoundaryIdentity::from_path(&[ElementId::Slot(0), ElementId::Slot(1)]);
        assert_ne!(first, second);
    }

    #[test]
    fn an_explicit_name_refines_the_position_rather_than_replacing_the_mechanism() {
        let positional = BoundaryIdentity::from_path(&[ElementId::Slot(0), ElementId::Slot(1)]);
        let named = BoundaryIdentity::from_path(&[ElementId::Slot(0), ElementId::from("panel")]);
        assert_ne!(positional, named);
        // The named one survives a move between sibling slots; the positional
        // one does not, which is exactly what `.id()` buys and all it buys.
        assert_eq!(
            named,
            BoundaryIdentity::from_path(&[ElementId::Slot(0), ElementId::from("panel")])
        );
    }

    #[test]
    fn the_window_root_is_the_boundary_of_an_empty_path() {
        assert_eq!(BoundaryIdentity::from_path(&[]), BoundaryId::ROOT);
    }

    #[test]
    fn a_boundary_identity_never_aliases_the_instance_key_from_the_same_path() {
        for path in [
            vec![ElementId::Slot(0)],
            vec![ElementId::from("scroller")],
            vec![ElementId::Slot(3), ElementId::Integer(9)],
        ] {
            assert_ne!(
                BoundaryIdentity::from_path(&path).as_raw(),
                InstanceKey::from_path(&path).as_raw()
            );
        }
    }

    #[test]
    fn each_boundary_gets_its_own_layer() {
        let first = BoundaryIdentity::layer_for_path(&[ElementId::Slot(0)]);
        let second = BoundaryIdentity::layer_for_path(&[ElementId::Slot(1)]);
        assert_ne!(first, second);
        assert_eq!(
            first,
            BoundaryIdentity::layer_for_path(&[ElementId::Slot(0)])
        );
        assert_eq!(
            BoundaryIdentity::layer_for_path(&[]),
            LayerId::from_key(LayerKey::untiled(BoundaryId::ROOT))
        );
    }
}
