//! `Description` — the cheap, per-frame value the frontend produces, and
//! `ElementId` — the identity segment it is addressed by.
//! See docs/gpu-native-architecture.md §2's diagram (the "Description
//! (per-frame, arena)" box) and R-N §2.1's three-lifetime table.
//!
//! Not in §3.1's literal file map — a deliberate addition, recorded in
//! `docs/phase-1-results.md`. §3.1 gives `reconcile/instance.rs` the retained
//! side of R-N §2.1's split (`ElementInstance`, `InstanceKey`) and never names
//! a home for the per-frame side, because in the legacy backend the two are
//! the same object (`Drawable<E>`). Separating them is the whole point of
//! Pillar I, so 2.0 gives each its own file.
//!
//! # Why this is not the legacy `Element` trait
//!
//! The legacy `Element` trait is tightly coupled to `Window` and `App` —
//! `request_layout`, `prepaint`, and `paint` all take them — and `wgpui-core`
//! cannot depend on either without dragging the whole legacy backend across
//! the crate boundary §3 draws. A `Description` is what is left once that
//! coupling is removed: identity, a type tag, a fingerprint, a layout style,
//! a scope flag, and children. Every element type in `wgpui-widgets` will
//! produce one of these; nothing here knows or cares which.

use crate::reconcile::diff_key::ReconcileKey;
use std::any::TypeId;
use std::sync::Arc;
use wgpui_layout::taffy_tree::LayoutStyle;

/// One segment of an element's path identity.
///
/// [`ElementId::Slot`] is *positional* identity (SFD §1.0) and is what makes
/// §4.0's "zero `.id()` touched anywhere" true: an element that never names
/// itself still gets a stable, cross-frame address from where it sits. An
/// explicit [`ElementId::Name`] or [`ElementId::Integer`] refines that so
/// identity survives reordering; it never gates it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ElementId {
    /// A name the author gave this element.
    Name(Arc<str>),
    /// An integer key, typically a list item's stable id.
    Integer(u64),
    /// Position among the parent's children. Assigned by the reconciler for
    /// every element that did not name itself, which is most of them.
    Slot(u32),
}

impl From<&str> for ElementId {
    fn from(name: &str) -> Self {
        ElementId::Name(Arc::from(name))
    }
}

impl From<String> for ElementId {
    fn from(name: String) -> Self {
        ElementId::Name(Arc::from(name.as_str()))
    }
}

impl From<u64> for ElementId {
    fn from(value: u64) -> Self {
        ElementId::Integer(value)
    }
}

/// The per-frame value a `render()` equivalent produces for one element.
///
/// Cheap to build and dropped at the end of the frame; everything expensive
/// (layout nodes, resolved bounds, emitted primitives) lives on the retained
/// [`crate::reconcile::instance::ElementInstance`] the reconciler matches this
/// against.
///
/// Consumed by reconciliation rather than borrowed by it: the retained
/// instance takes ownership of this frame's fingerprint, and a `Box<dyn
/// ReconcileKey>` cannot be cloned out of a shared borrow without adding a
/// clone bound every element author would then have to satisfy. Consuming is
/// also what actually happens — a description is a per-frame value that is
/// dropped at the end of the frame either way.
pub struct Description {
    pub(crate) element_id: Option<ElementId>,
    pub(crate) type_id: TypeId,
    pub(crate) type_name: &'static str,
    pub(crate) diff_key: Option<Box<dyn ReconcileKey>>,
    pub(crate) uncached: bool,
    pub(crate) layout_style: LayoutStyle,
    pub(crate) children: Vec<Description>,
}

impl std::fmt::Debug for Description {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Description")
            .field("element_id", &self.element_id)
            .field("type_name", &self.type_name)
            .field("has_diff_key", &self.diff_key.is_some())
            .field("uncached", &self.uncached)
            .field("children", &self.children.len())
            .finish()
    }
}

impl Description {
    /// A description of an element of type `T`, with no fingerprint.
    ///
    /// No fingerprint means "assume changed, rebuild" — R-N §2.3's permissive
    /// default, correct for a third-party element whose purity cannot be
    /// proven from outside. Note what it does *not* mean: the element still
    /// gets an instance record and still keeps its layout node where it can.
    /// Removing the record entirely is `.uncached()`'s job (§4.2), and keeping
    /// the two distinct is exactly what Phase 1's third gate checks.
    pub fn new<T: 'static>() -> Self {
        Self {
            element_id: None,
            type_id: TypeId::of::<T>(),
            type_name: std::any::type_name::<T>(),
            diff_key: None,
            uncached: false,
            layout_style: LayoutStyle::default(),
            children: Vec::new(),
        }
    }

    /// Give this element an explicit identity, so it keeps its instance across
    /// a sibling reorder rather than being matched positionally.
    pub fn id(mut self, element_id: impl Into<ElementId>) -> Self {
        self.element_id = Some(element_id.into());
        self
    }

    /// Attach this frame's fingerprint.
    pub fn diff_key(mut self, key: impl ReconcileKey) -> Self {
        self.diff_key = Some(Box::new(key));
        self
    }

    /// Opt this element and its whole subtree out of reconciliation (§4.2).
    ///
    /// No instance record is allocated for anything inside, no fingerprint is
    /// retained, and no comparison runs — strictly less bookkeeping than
    /// reconciling and always losing, not merely a skip. State keyed by path
    /// and type is untouched: see
    /// [`crate::reconcile::state::ElementStateStore`].
    pub fn uncached(mut self) -> Self {
        self.uncached = true;
        self
    }

    /// Set the style this element's layout node is laid out with.
    pub fn style(mut self, style: LayoutStyle) -> Self {
        self.layout_style = style;
        self
    }

    /// Append a child.
    pub fn child(mut self, child: Description) -> Self {
        self.children.push(child);
        self
    }

    /// Append several children.
    pub fn children(mut self, children: impl IntoIterator<Item = Description>) -> Self {
        self.children.extend(children);
        self
    }

    /// The explicit identity this element was given, if any.
    pub fn element_id(&self) -> Option<&ElementId> {
        self.element_id.as_ref()
    }

    /// The element's Rust type, which must match across frames for an instance
    /// to be reused.
    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    /// The element's Rust type name, carried for diagnostics only.
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// This frame's fingerprint, if the element supplied one.
    pub fn key(&self) -> Option<&dyn ReconcileKey> {
        self.diff_key.as_deref()
    }

    /// Whether this element opted its subtree out of reconciliation.
    pub fn is_uncached(&self) -> bool {
        self.uncached
    }

    /// The style this element's layout node is laid out with.
    pub fn layout_style(&self) -> &LayoutStyle {
        &self.layout_style
    }

    /// This element's children, in order.
    pub fn child_descriptions(&self) -> &[Description] {
        &self.children
    }

    /// Total nodes in this description subtree, including this one.
    pub fn node_count(&self) -> usize {
        1 + self
            .children
            .iter()
            .map(Description::node_count)
            .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconcile::diff_key::AlwaysDirty;

    struct Panel;

    #[test]
    fn a_bare_description_names_nothing_and_fingerprints_nothing() {
        let description = Description::new::<Panel>();
        assert!(description.element_id().is_none());
        assert!(description.key().is_none());
        assert!(!description.is_uncached());
        assert_eq!(description.node_count(), 1);
    }

    #[test]
    fn children_are_kept_in_order() {
        let description = Description::new::<Panel>()
            .child(Description::new::<Panel>().id("first"))
            .child(Description::new::<Panel>().id("second"));
        let ids: Vec<Option<&ElementId>> = description
            .child_descriptions()
            .iter()
            .map(Description::element_id)
            .collect();
        assert_eq!(ids[0], Some(&ElementId::Name(Arc::from("first"))));
        assert_eq!(ids[1], Some(&ElementId::Name(Arc::from("second"))));
        assert_eq!(description.node_count(), 3);
    }

    #[test]
    fn node_count_walks_the_whole_subtree() {
        let leaf = || Description::new::<Panel>().diff_key(AlwaysDirty);
        let description = Description::new::<Panel>()
            .child(Description::new::<Panel>().child(leaf()).child(leaf()))
            .child(leaf());
        assert_eq!(description.node_count(), 5);
    }

    #[test]
    fn element_ids_from_different_sources_are_distinct() {
        assert_ne!(ElementId::from("7"), ElementId::from(7u64));
        assert_ne!(ElementId::from(7u64), ElementId::Slot(7));
        assert_eq!(ElementId::from("row"), ElementId::from(String::from("row")));
    }
}
