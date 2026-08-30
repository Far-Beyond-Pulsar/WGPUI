//! Retained per-element state: `ElementInstance` + reconciliation (#92).
//!
//! Phase 4 (`layer.rs`) gave the renderer a retained, independently
//! invalidated unit of caching — but a `.layer()` div still had exactly one
//! bit of memory about its content: "did the owning view get notified since
//! last frame?" Any notify forced the *entire* subtree back through
//! `request_layout`/`prepaint`/`paint`, even when the freshly-rebuilt
//! description turned out to be identical to what was already there. That gap
//! is named directly in [`crate::window::Window::with_retained_layer`]'s doc
//! comment: "reconciliation is #92... a rebuilt description has to be assumed
//! different."
//!
//! This module closes it. An [`ElementInstance`] is last frame's retained
//! record for one element, keyed by [`InstanceKey`] — a positional-or-named
//! address, not unlike [`crate::layer::LayerKey`], except addressed *within*
//! a layer rather than within the whole window (see [`Window::instance_id_stack`]
//! for why the two are kept separate). [`Element::diff_key`] gives an element
//! type a cheap, owned, arena-free fingerprint of "what I looked like this
//! frame"; comparing it against the retained one is what decides whether
//! `Div`'s child loop can skip a child's `prepaint`/`paint` entirely and reuse
//! what was recorded last time.
//!
//! # What this phase does not do
//!
//! `request_layout` — and therefore `render()` for nested views — still runs
//! unconditionally for every element, every frame, including reconciled
//! ones; description building was never what this phase (or #93) skips. What
//! gets skipped here is `prepaint` and `paint`: text shaping, primitive
//! emission, `BoundsTree` insertion, hitbox/dispatch-node registration —
//! reused via the same bounds-checked index-range-replay mechanism
//! `AnyViewState` (`view.rs`) already uses for whole views, just keyed at
//! element granularity instead.
//!
//! Taffy node reuse itself — an `ElementInstance` retaining its `layout:
//! LayoutId` across frames, and `TaffyLayoutEngine` no longer being
//! unconditionally cleared — is phase 8 (#93), layered on top of the
//! `InstanceKey`/`ReconcileKey` machinery this module defines. See
//! `Window::request_layout_or_reuse` and `TaffyLayoutEngine::end_frame`.
//!
//! Reconciliation is also scoped to content painted inside a `.layer()` div's
//! subtree. `ElementInstance`s live in [`crate::layer::Layer::instances`], so
//! their memory is bounded by the same mark-and-sweep eviction that already
//! bounds a layer's retained primitives — instances are owned by layers and
//! die with them, exactly as `Window::evict_stale_layers`'s doc comment
//! anticipated before this module existed. Content with no `.layer()`
//! ancestor gets no benefit from this phase; it rebuilds every frame exactly
//! as it does today.

use crate::{
    Bounds, ContentMask, ElementId, EntityId, Invalidation, LayoutId, PaintIndex, Pixels,
    PrepaintStateIndex,
};
use crate::layer::LayerItem;
use collections::FxHashSet;
use std::any::Any;
use std::hash::{Hash, Hasher};
use std::ops::Range;

/// The stable address of one retained element within its owning layer.
///
/// Derived from the path of [`ElementId`]s on [`crate::window::Window::instance_id_stack`]
/// at the point the element was visited — a real `ElementId` for an element
/// that called `.id(...)`, or a synthetic `ElementId::InstanceSlot` for one
/// that didn't (the common case: bare `div()`, all of `Text`). This is the
/// same hash-the-path-of-ids technique [`crate::layer::LayerKey`] uses for
/// exactly the same reason: identity that survives across frames without
/// requiring every element to be named.
///
/// Two elements at the same position under two different parents get
/// different keys because the whole path is hashed, not just the last
/// segment — see the `distinct_under_different_parents` test below.
///
/// Deliberately **not** typed by the element's Rust type: a position can hold
/// a `Div` one frame and an `Img` the next (`if cond { div() } else { img() }`),
/// and `InstanceKey` alone can't and shouldn't tell them apart — that is
/// [`ReconcileKey::compare`]'s job, via a failed downcast, so a type change is
/// a rebuild rather than a key collision.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstanceKey(u64);

impl InstanceKey {
    /// Derive the key for the element addressed by `path`, the current
    /// contents of `Window::instance_id_stack`.
    pub(crate) fn from_path(path: &[ElementId]) -> Self {
        let mut hasher = collections::FxHasher::default();
        path.hash(&mut hasher);
        // Reserve 0 the same way `LayerKey` does, so a defaulted key is never
        // mistaken for a live instance.
        InstanceKey(hasher.finish() | 1)
    }
}

/// A cheap, owned, arena-free fingerprint of one frame's element description,
/// compared against the previous frame's fingerprint for the same
/// [`InstanceKey`] to decide whether `prepaint`/`paint` may be skipped.
///
/// Implementations are small plain values holding only what their element's
/// `diff_key` chooses to snapshot — never the arena-allocated description
/// itself (its children are `AnyElement`s bump-allocated into an arena that
/// is cleared every frame; retaining that verbatim across frames would retain
/// dangling pointers). See `Div`'s implementation in `elements/div.rs` for
/// the canonical shape: a cloned `StyleRefinement` plus the ordered list of
/// child `InstanceKey`s.
pub trait ReconcileKey: Any {
    /// Compare against last frame's key for the same `InstanceKey`.
    /// `Invalidation::empty()` means the element is fully reusable.
    ///
    /// Implementations must downcast `previous` to `Self` first and treat a
    /// failed downcast as `Invalidation::all()` — a position that held a
    /// different element type last frame is exactly the case that must never
    /// be reused, and a failed downcast is how that shows up here.
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation;

    /// Downcasting helper. `dyn ReconcileKey: Any` alone isn't enough to call
    /// `Any`'s own methods on a `&dyn ReconcileKey` without this.
    fn as_any(&self) -> &dyn Any;
}

/// Retained state for one element, from the last frame it was visited.
///
/// Lives in [`crate::layer::Layer::instances`], keyed by [`InstanceKey`].
pub(crate) struct ElementInstance {
    /// Last frame's fingerprint, compared against this frame's fresh one.
    pub diff_key: Box<dyn ReconcileKey>,
    /// This element's own Taffy node (#93). Valid to hand back to
    /// `TaffyLayoutEngine::reuse` for as long as this entry survives — which
    /// is exactly as long as the node itself does, since both die together:
    /// a rebuild that replaces this entry also creates a fresh node (the old
    /// one simply goes untouched and is swept at end of frame, see
    /// `TaffyLayoutEngine::end_frame`), and a layer eviction that clears this
    /// entry leaves the node to be swept the same way.
    pub layout: LayoutId,
    /// Resolved bounds at last prepaint. Reuse requires an exact match — no
    /// partial-translate reuse in this phase; that is `TRANSFORM`-axis
    /// territory a future phase can add without changing this shape.
    pub bounds: Bounds<Pixels>,
    pub content_mask: ContentMask<Pixels>,
    /// Bracket into `rendered_frame`'s arrays for everything `prepaint`
    /// registered: hitboxes, tooltip requests, accessed element states, the
    /// dispatch subtree, deferred draws, shaped text. Replayed via
    /// `Window::reuse_prepaint`, bounds-checked via `Window::invalid_reuse_range`
    /// — the exact mechanism `AnyViewState` already uses for whole views.
    pub prepaint_range: Range<PrepaintStateIndex>,
    /// Bracket into `rendered_frame`'s arrays for everything `paint`
    /// registered *except* primitives: cursor styles, input handlers, mouse
    /// listeners, accessed element states, tab stops, shaped text. Replayed
    /// via `Window::reuse_paint_except_scene`.
    pub paint_range: Range<PaintIndex>,
    /// This instance's own retained primitives (and any nested `.layer()`
    /// references within its subtree), in paint order, carrying the
    /// layer-local draw orders they were recorded with. Replayed via
    /// `Window::replay_instance_items`, which re-emits each one through
    /// `Scene::push_retained` — no `BoundsTree` insert, no re-derivation of z.
    ///
    /// An owned `Vec` rather than a range into the owning layer's `items`
    /// (which the design doc's initial sketch used) so reuse never depends on
    /// the layer's item list not having been overwritten yet by the time this
    /// instance is replayed. This is Pillar I's version of the primitive; a
    /// per-layer slab range (Pillar III, phase 9+) can replace it later
    /// without changing anything about how reconciliation decides to reuse.
    pub items: Vec<LayerItem>,
    /// Entities read while this element (and its subtree) last rendered. An
    /// invalidation naming any of these forces a rebuild even if `diff_key`
    /// compares equal — mirrors `AnyViewState::accessed_entities` exactly.
    pub accessed_entities: FxHashSet<EntityId>,
}

/// Whether `.layer()` subtrees reconcile at element granularity.
///
/// `WGPUI_INSTANCES=0` makes every element behave exactly as it did before
/// this module existed: `Div`'s child loop always calls `prepaint`/`paint`,
/// and no `ElementInstance` is ever consulted. Following the `WGPUI_LAYERS`
/// precedent (`layer.rs`) — this phase changes what a `.layer()` subtree
/// reuses, so the old always-rebuild path stays reachable without a rebuild
/// until a later phase removes it.
///
/// Read once, at first use.
pub(crate) fn instances_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("WGPUI_INSTANCES")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(true)
    });
    *ENABLED
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ElementId;

    fn path(segments: &[ElementId]) -> Vec<ElementId> {
        segments.to_vec()
    }

    #[test]
    fn instance_key_is_stable_across_frames() {
        let a = InstanceKey::from_path(&path(&[
            ElementId::Name("root".into()),
            ElementId::InstanceSlot(0),
        ]));
        let b = InstanceKey::from_path(&path(&[
            ElementId::Name("root".into()),
            ElementId::InstanceSlot(0),
        ]));
        assert_eq!(a, b, "the same path must produce the same key every frame");
    }

    #[test]
    fn instance_key_distinguishes_positional_siblings() {
        let first = InstanceKey::from_path(&path(&[
            ElementId::Name("list".into()),
            ElementId::InstanceSlot(0),
        ]));
        let second = InstanceKey::from_path(&path(&[
            ElementId::Name("list".into()),
            ElementId::InstanceSlot(1),
        ]));
        assert_ne!(
            first, second,
            "two positional children of the same parent must not collide"
        );
    }

    #[test]
    fn instance_key_distinct_under_different_parents() {
        let under_a = InstanceKey::from_path(&path(&[
            ElementId::Name("a".into()),
            ElementId::InstanceSlot(0),
        ]));
        let under_b = InstanceKey::from_path(&path(&[
            ElementId::Name("b".into()),
            ElementId::InstanceSlot(0),
        ]));
        assert_ne!(
            under_a, under_b,
            "the same slot under a different parent is a different instance"
        );
    }

    #[test]
    fn instance_key_is_never_zero() {
        for p in [
            vec![ElementId::InstanceSlot(0)],
            vec![ElementId::Name("a".into())],
            vec![],
        ] {
            assert_ne!(InstanceKey::from_path(&p).0, 0);
        }
    }

    /// A minimal `ReconcileKey` used only to exercise the downcast-or-differ
    /// contract every real implementation (Div, Text, Img, Svg) must follow.
    #[derive(PartialEq)]
    struct TestKey(u32);

    impl ReconcileKey for TestKey {
        fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
            match previous.as_any().downcast_ref::<TestKey>() {
                Some(prev) if prev == self => Invalidation::empty(),
                Some(_) => Invalidation::DISPLAY,
                None => Invalidation::all(),
            }
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    struct OtherKey;
    impl ReconcileKey for OtherKey {
        fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
            match previous.as_any().downcast_ref::<OtherKey>() {
                Some(_) => Invalidation::empty(),
                None => Invalidation::all(),
            }
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn reconcile_key_equal_values_compare_empty() {
        let a = TestKey(1);
        let b = TestKey(1);
        assert_eq!(a.compare(&b), Invalidation::empty());
    }

    #[test]
    fn reconcile_key_different_values_compare_non_empty() {
        let a = TestKey(1);
        let b = TestKey(2);
        assert_ne!(a.compare(&b), Invalidation::empty());
    }

    #[test]
    fn reconcile_key_type_mismatch_is_full_invalidation() {
        let a = TestKey(1);
        let b = OtherKey;
        assert_eq!(
            a.compare(&b),
            Invalidation::all(),
            "a position that held a different element type last frame must never be reused"
        );
    }
}
