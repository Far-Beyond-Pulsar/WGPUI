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

use crate::boundary::policy::BoundaryPolicy;
use crate::patch::emit::Emit;
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

/// Text content produced by a raw string child.
///
/// The content remains renderer-independent in `wgpui-core`; the renderer
/// that owns fonts and atlas pages resolves it into glyph primitives before
/// layout. Keeping the string here gives every backend the same frontend
/// contract without making the core crate depend on a text implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawText {
    value: Arc<str>,
}

impl RawText {
    /// Construct raw text from an owned or borrowed string.
    pub fn new(value: impl Into<Arc<str>>) -> Self {
        Self {
            value: value.into(),
        }
    }

    /// The UTF-8 contents of this text node.
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Share the text without copying it.
    pub fn shared_value(&self) -> Arc<str> {
        Arc::clone(&self.value)
    }
}

/// Stable fingerprint for a raw string's content.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawTextKey {
    value: Arc<str>,
}

impl RawTextKey {
    fn new(value: Arc<str>) -> Self {
        Self { value }
    }
}

impl crate::reconcile::diff_key::ReconcileKey for RawTextKey {
    fn compare(
        &self,
        previous: &dyn crate::reconcile::diff_key::ReconcileKey,
    ) -> crate::invalidation::axes::Invalidation {
        crate::reconcile::diff_key::compare_by_equality(
            self,
            previous,
            crate::invalidation::axes::Invalidation::LAYOUT
                | crate::invalidation::axes::Invalidation::DISPLAY,
        )
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
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

impl From<(&str, usize)> for ElementId {
    fn from((name, index): (&str, usize)) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        name.hash(&mut hasher);
        index.hash(&mut hasher);
        ElementId::Integer(hasher.finish())
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
    pub(crate) boundary: Option<BoundaryPolicy>,
    pub(crate) scroll_offset: [f32; 2],
    pub(crate) emitter: Option<Box<dyn Emit>>,
    pub(crate) layout_style: LayoutStyle,
    pub(crate) children: Vec<Description>,
    pub(crate) clip_children: bool,
    pub(crate) raw_text: Option<RawText>,
}

impl std::fmt::Debug for Description {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Description")
            .field("element_id", &self.element_id)
            .field("type_name", &self.type_name)
            .field("has_diff_key", &self.diff_key.is_some())
            .field("uncached", &self.uncached)
            .field("boundary", &self.boundary)
            .field("scroll_offset", &self.scroll_offset)
            .field("has_emitter", &self.emitter.is_some())
            .field("children", &self.children.len())
            .field("clip_children", &self.clip_children)
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
            boundary: None,
            scroll_offset: [0.0, 0.0],
            emitter: None,
            layout_style: LayoutStyle::default(),
            children: Vec::new(),
            clip_children: false,
            raw_text: None,
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

    /// Make this element a compositing boundary (§4.1).
    ///
    /// Takes no arguments, and that is the whole design: the subtree beneath it
    /// is already being reconciled (§4.0), so this asks only for independent
    /// compositing — its own layer, its own retention decision, and the ability
    /// to resolve a scroll to a transform rather than to re-emitted content.
    /// No `.id()` is needed either: the boundary's identity comes from its
    /// position (SFD §1.0), via
    /// [`crate::boundary::identity::BoundaryIdentity`]. A forgotten `.id()`
    /// costs the boundary its identity only across a sibling reorder, and even
    /// then costs one rebuilt frame rather than correctness.
    pub fn boundary(self) -> Self {
        self.boundary_with_policy(BoundaryPolicy::default())
    }

    /// Make this element a compositing boundary with tuning.
    ///
    /// [`BoundaryPolicy`] never affects whether the boundary is considered
    /// dirty — only how a boundary already known to be dirty is rasterized and
    /// buffered.
    pub fn boundary_with_policy(mut self, policy: BoundaryPolicy) -> Self {
        self.boundary = Some(policy);
        self
    }

    /// Displace this element's children by `offset`.
    ///
    /// This is a scroll or pan offset, expressed as what it does rather than as
    /// a scroll position, because the two paths that consume it consume it
    /// identically: a boundary installs it as its layer's transform and leaves
    /// its children's emitted positions alone, while an ordinary element folds
    /// it into those positions. That symmetry is what makes `.boundary()` a
    /// pure optimization — removing it changes which of the two happens and
    /// nothing else about the frame.
    ///
    /// Deliberately not part of any `diff_key`: SFD §1.1's whole finding is
    /// that a scroll container's key must cover everything *except* its offset,
    /// and here it cannot accidentally be included because the offset is not
    /// something an element's fingerprint can see.
    pub fn scroll_offset(mut self, offset: [f32; 2]) -> Self {
        self.scroll_offset = offset;
        self
    }

    /// Give this element something to emit into the scene, given its resolved
    /// bounds.
    ///
    /// See [`Emit`] for the shape and for why this is optional: an element that
    /// only groups children — most of a real tree — emits nothing itself.
    pub fn emit(mut self, emitter: impl Emit) -> Self {
        self.emitter = Some(Box::new(emitter));
        self
    }

    /// Make this description a raw text node.
    pub fn raw_text(value: impl Into<Arc<str>>) -> Self {
        let value = value.into();
        Self::new::<RawText>()
            .diff_key(RawTextKey::new(Arc::clone(&value)))
            .with_raw_text(RawText::new(value))
    }

    fn with_raw_text(mut self, raw_text: RawText) -> Self {
        self.raw_text = Some(raw_text);
        self
    }

    /// Take unresolved raw text so a renderer can materialize it.
    pub fn take_raw_text(&mut self) -> Option<RawText> {
        self.raw_text.take()
    }

    /// Replace automatic dimensions with measured text dimensions.
    pub fn set_intrinsic_size(&mut self, width: f32, height: f32) {
        if self.layout_style.size.width == wgpui_layout::taffy_tree::Dimension::auto() {
            self.layout_style.size.width = wgpui_layout::taffy_tree::Dimension::length(width);
        }
        if self.layout_style.size.height == wgpui_layout::taffy_tree::Dimension::auto() {
            self.layout_style.size.height = wgpui_layout::taffy_tree::Dimension::length(height);
        }
    }

    /// Attach the renderer-produced text emitter after raw text has been
    /// shaped and its glyphs have been assigned atlas tiles.
    pub fn set_text_emitter(&mut self, emitter: impl Emit) {
        self.emitter = Some(Box::new(emitter));
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

    /// Clip this element's descendants to its resolved bounds. This is
    /// retained metadata consumed by the emitter; it is not a paint callback
    /// and therefore remains valid when the subtree is reused.
    pub fn clip_children(mut self) -> Self {
        self.clip_children = true;
        self
    }

    pub fn clips_children(&self) -> bool {
        self.clip_children
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

    /// Whether this element declared itself a compositing boundary.
    pub fn is_boundary(&self) -> bool {
        self.boundary.is_some()
    }

    /// The tuning this element's boundary was declared with, if any.
    ///
    /// `Some(BoundaryPolicy::default())` is what a bare `.boundary()` produces,
    /// which is how a test can check "zero policy arguments" mechanically
    /// rather than by reading the call site.
    pub fn boundary_policy(&self) -> Option<BoundaryPolicy> {
        self.boundary
    }

    /// The displacement this element applies to its children.
    pub fn scroll_offset_of(&self) -> [f32; 2] {
        self.scroll_offset
    }

    /// Whether this element emits anything of its own.
    pub fn emits(&self) -> bool {
        self.emitter.is_some()
    }

    /// The style this element's layout node is laid out with.
    pub fn layout_style(&self) -> &LayoutStyle {
        &self.layout_style
    }

    /// This element's children, in order.
    pub fn child_descriptions(&self) -> &[Description] {
        &self.children
    }

    /// Mutable children for a backend materialization pass.
    pub fn child_descriptions_mut(&mut self) -> &mut [Description] {
        &mut self.children
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
