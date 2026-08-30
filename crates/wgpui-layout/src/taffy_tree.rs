//! Persistent `TaffyTree` wrapper — today's `src/taffy.rs`, made ambient
//! per §4.0. See docs/gpu-native-architecture.md §3.2.
//!
//! # What "made ambient" changes
//!
//! The legacy engine already persists nodes across frames (R-N §2.5 / phase 8)
//! using a touched-set sweep, and that mechanism is right and is kept. What
//! changes is the *condition* under which reuse is offered: the legacy path
//! only reaches `reuse` for content inside a `.layer()` subtree, so an
//! ordinary panel's nodes are recreated every frame no matter how obviously
//! unchanged they are. Here there is no such gate — [`LayoutTree`] has no
//! concept of a layer or a boundary at all, and the reconciler that drives it
//! (`wgpui-core::reconcile`) offers reuse for every element in the tree by
//! construction. §4.0 is enforced by an absence, which is the only way an
//! ambient default stays ambient.
//!
//! # Errors, not panics
//!
//! The legacy engine funnels every Taffy result through one `expect`. This
//! wrapper propagates instead: a wrong node id is a bookkeeping bug, and the
//! discipline the whole 2.0 reconciliation story rests on (R-N §2.2: "a
//! mismatch causes a subtree rebuild — one slow frame, never incorrect
//! output") only holds if a miss is recoverable.

use std::collections::HashSet;
use taffy::TaffyTree;
pub use taffy::geometry::{Line, Size as TaffySize};
pub use taffy::style::AvailableSpace;
pub use taffy::style_helpers::FromFr;
use taffy::tree::NodeId;

/// The style a layout node is laid out with.
///
/// Re-exported from `taffy` rather than wrapped: §10 rules a new layout DSL
/// out of scope, `wgpui-widgets` owns the Tailwind-style surface that
/// *produces* one of these (§7), and an intermediate copy here would be a
/// third representation of the same thing with nothing to say for itself.
pub type LayoutStyle = taffy::style::Style;

/// A width/height pair, as Taffy spells one.
pub type LayoutSize<T> = TaffySize<T>;

/// A top/right/bottom/left quadruple, as Taffy spells one.
///
/// Named `LayoutSides` rather than `LayoutRect` because that name is already
/// taken by the *computed* rectangle below, and Taffy's own `Rect` is neither —
/// it is the four-sided value `padding`, `margin`, `border` and `inset` are all
/// expressed as.
pub type LayoutSides<T> = taffy::geometry::Rect<T>;

/// Every style enum and length type `wgpui-widgets`' Tailwind surface has to
/// name in order to build a [`LayoutStyle`].
///
/// Re-exported for the reason the module doc already gives for `LayoutStyle`
/// itself: §10 rules a new layout DSL out of scope and §3.2 does not intend the
/// `taffy` dependency to leak past this crate, so anything a caller must spell
/// is spelled here once. Phase 6.6 widened this list from three names to the
/// set a real `div()` builder needs; nothing about the policy changed.
pub use taffy::style::{
    AlignContent, AlignItems, BoxSizing, Dimension, Display, FlexDirection, FlexWrap,
    GridPlacement, GridTemplateComponent, LengthPercentage, LengthPercentageAuto, Overflow,
    Position, TrackSizingFunction,
};

/// The available space for a subtree with a known width and height.
///
/// A convenience for the common case, so a caller with two numbers does not
/// have to name [`AvailableSpace`] itself — [`LayoutTree::compute_layout`]'s
/// signature already obliges every caller to name `taffy`'s types, which is a
/// leak §3.2 does not intend; this and the re-exports above are the narrowest
/// way to close it without wrapping the style type §10 rules out of scope.
pub const fn definite(width: f32, height: f32) -> LayoutSize<AvailableSpace> {
    TaffySize {
        width: AvailableSpace::Definite(width),
        height: AvailableSpace::Definite(height),
    }
}

/// A retained layout node's identity, stable for as long as the node lives.
///
/// Gate #2 of Phase 1 is, in this crate's terms, that a clean element's
/// `LayoutNodeId` is bit-identical frame over frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayoutNodeId(u64);

impl LayoutNodeId {
    fn from_taffy(node: NodeId) -> Self {
        LayoutNodeId(node.into())
    }

    fn to_taffy(self) -> NodeId {
        NodeId::from(self.0)
    }

    /// The raw identity, for debug output and test assertions.
    pub const fn as_raw(self) -> u64 {
        self.0
    }

    /// Wrap a raw identity.
    ///
    /// Safe to call with any value: every operation that touches the tree
    /// checks liveness first, so an id that no tree ever produced is simply
    /// never live and every call against it reports a miss.
    pub const fn from_raw(raw: u64) -> Self {
        LayoutNodeId(raw)
    }
}

/// A computed node rectangle, in the layout tree's own coordinate space.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LayoutRect {
    /// Distance from the parent's left edge.
    pub x: f32,
    /// Distance from the parent's top edge.
    pub y: f32,
    /// Computed width.
    pub width: f32,
    /// Computed height.
    pub height: f32,
}

/// Something went wrong addressing or laying out a node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    /// The node is not (or is no longer) live in this tree.
    UnknownNode(LayoutNodeId),
    /// Taffy itself rejected the operation.
    Taffy(String),
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayoutError::UnknownNode(node) => {
                write!(formatter, "layout node {} is not live", node.as_raw())
            }
            LayoutError::Taffy(message) => write!(formatter, "taffy: {message}"),
        }
    }
}

impl std::error::Error for LayoutError {}

impl From<taffy::TaffyError> for LayoutError {
    fn from(error: taffy::TaffyError) -> Self {
        LayoutError::Taffy(error.to_string())
    }
}

/// How much layout work a frame actually did — the direct measurement behind
/// §4.0's claim that an unchanged subtree keeps its nodes.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct LayoutFrameStats {
    /// Nodes created this frame.
    pub nodes_created: usize,
    /// Nodes reused from a previous frame.
    pub nodes_reused: usize,
    /// Nodes swept at the end of the previous frame.
    pub nodes_swept: usize,
}

/// A Taffy tree whose nodes persist across frames.
///
/// Every element calls exactly one of [`LayoutTree::request_layout`] or
/// [`LayoutTree::reuse`] per frame, so a node absent from the touched set at
/// [`LayoutTree::end_frame`] is unambiguously gone from this frame's tree
/// rather than merely not visited yet.
#[derive(Debug)]
pub struct LayoutTree {
    tree: TaffyTree<()>,
    live: HashSet<NodeId>,
    touched: HashSet<NodeId>,
    stats: LayoutFrameStats,
}

impl Default for LayoutTree {
    fn default() -> Self {
        Self::new()
    }
}

impl LayoutTree {
    /// An empty tree with rounding enabled, matching the legacy engine.
    pub fn new() -> Self {
        let mut tree = TaffyTree::new();
        tree.enable_rounding();
        Self {
            tree,
            live: HashSet::new(),
            touched: HashSet::new(),
            stats: LayoutFrameStats::default(),
        }
    }

    /// Start a frame: clears the per-frame counters, not the tree.
    pub fn begin_frame(&mut self) {
        self.stats = LayoutFrameStats::default();
    }

    /// Create a node with `style` and `children`, marking it present in this
    /// frame's tree.
    pub fn request_layout(
        &mut self,
        style: LayoutStyle,
        children: &[LayoutNodeId],
    ) -> Result<LayoutNodeId, LayoutError> {
        let node = if children.is_empty() {
            self.tree.new_leaf(style)?
        } else {
            let taffy_children: Vec<NodeId> =
                children.iter().map(|child| child.to_taffy()).collect();
            self.tree.new_with_children(style, &taffy_children)?
        };
        self.live.insert(node);
        self.touched.insert(node);
        self.stats.nodes_created += 1;
        Ok(LayoutNodeId::from_taffy(node))
    }

    /// Mark a retained node as present in this frame's tree without
    /// recreating it — the layout counterpart of reconciliation's
    /// `prepaint`/`paint` skip.
    ///
    /// Returns `false` when the node is not (or is no longer) live, so a
    /// caller's correct response is to create a fresh one: a miss is a
    /// rebuild, never a crash.
    pub fn reuse(&mut self, node: LayoutNodeId) -> bool {
        let taffy_node = node.to_taffy();
        if !self.live.contains(&taffy_node) {
            return false;
        }
        if self.touched.insert(taffy_node) {
            self.stats.nodes_reused += 1;
        }
        true
    }

    /// Replace a live node's style. Taffy's own dirty propagation decides what
    /// that invalidates.
    pub fn set_style(&mut self, node: LayoutNodeId, style: LayoutStyle) -> Result<(), LayoutError> {
        self.require_live(node)?;
        self.tree.set_style(node.to_taffy(), style)?;
        Ok(())
    }

    /// Replace a live node's child list.
    pub fn set_children(
        &mut self,
        node: LayoutNodeId,
        children: &[LayoutNodeId],
    ) -> Result<(), LayoutError> {
        self.require_live(node)?;
        let taffy_children: Vec<NodeId> = children.iter().map(|child| child.to_taffy()).collect();
        self.tree.set_children(node.to_taffy(), &taffy_children)?;
        Ok(())
    }

    /// A live node's current child list.
    pub fn children(&self, node: LayoutNodeId) -> Result<Vec<LayoutNodeId>, LayoutError> {
        self.require_live(node)?;
        Ok(self
            .tree
            .children(node.to_taffy())?
            .into_iter()
            .map(LayoutNodeId::from_taffy)
            .collect())
    }

    /// Lay out the subtree rooted at `root` inside `available`.
    pub fn compute_layout(
        &mut self,
        root: LayoutNodeId,
        available: TaffySize<AvailableSpace>,
    ) -> Result<(), LayoutError> {
        self.require_live(root)?;
        self.tree.compute_layout(root.to_taffy(), available)?;
        Ok(())
    }

    /// A node's computed rectangle, relative to its parent.
    pub fn layout_of(&self, node: LayoutNodeId) -> Result<LayoutRect, LayoutError> {
        self.require_live(node)?;
        let layout = self.tree.layout(node.to_taffy())?;
        Ok(LayoutRect {
            x: layout.location.x,
            y: layout.location.y,
            width: layout.size.width,
            height: layout.size.height,
        })
    }

    /// Remove every node not touched this frame.
    ///
    /// Safe regardless of removal order among the swept set: `TaffyTree`
    /// detaches a node from its parent and clears its children's parent
    /// pointers through checked lookups, so a parent and child both present in
    /// one sweep do not race each other.
    pub fn end_frame(&mut self) -> usize {
        let orphaned: Vec<NodeId> = self
            .live
            .iter()
            .copied()
            .filter(|node| !self.touched.contains(node))
            .collect();

        let mut swept = 0;
        for node in orphaned {
            if let Err(error) = self.tree.remove(node) {
                // `live` disagreeing with the tree itself is defensive, not
                // expected. Dropping our own bookkeeping entry anyway keeps a
                // disagreement a leak of one node rather than a panic on the
                // frame path, and logging is not available to this crate.
                debug_assert!(false, "sweeping an untouched node failed: {error}");
            }
            swept += 1;
            self.live.remove(&node);
        }
        self.touched.clear();
        self.stats.nodes_swept = swept;
        swept
    }

    /// Whether a node is live in this tree.
    pub fn is_live(&self, node: LayoutNodeId) -> bool {
        self.live.contains(&node.to_taffy())
    }

    /// How many nodes are live.
    pub fn live_node_count(&self) -> usize {
        self.live.len()
    }

    /// This frame's creation/reuse/sweep counters.
    pub fn stats(&self) -> LayoutFrameStats {
        self.stats
    }

    fn require_live(&self, node: LayoutNodeId) -> Result<(), LayoutError> {
        if self.live.contains(&node.to_taffy()) {
            Ok(())
        } else {
            Err(LayoutError::UnknownNode(node))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use taffy::style::Dimension;

    fn sized(width: f32, height: f32) -> LayoutStyle {
        LayoutStyle {
            size: TaffySize {
                width: Dimension::length(width),
                height: Dimension::length(height),
            },
            ..LayoutStyle::default()
        }
    }

    #[test]
    fn a_reused_node_keeps_its_identity_across_frames() -> Result<(), LayoutError> {
        let mut tree = LayoutTree::new();
        tree.begin_frame();
        let node = tree.request_layout(sized(10.0, 10.0), &[])?;
        tree.end_frame();

        tree.begin_frame();
        assert!(tree.reuse(node));
        assert_eq!(tree.end_frame(), 0);
        assert!(tree.is_live(node));
        assert_eq!(tree.live_node_count(), 1);
        Ok(())
    }

    #[test]
    fn an_untouched_node_is_swept() -> Result<(), LayoutError> {
        let mut tree = LayoutTree::new();
        tree.begin_frame();
        let node = tree.request_layout(sized(10.0, 10.0), &[])?;
        tree.end_frame();

        tree.begin_frame();
        assert_eq!(tree.end_frame(), 1);
        assert!(!tree.is_live(node));
        assert!(!tree.reuse(node), "a swept node must not be reusable");
        Ok(())
    }

    #[test]
    fn reusing_an_unknown_node_reports_a_miss_rather_than_panicking() {
        let mut tree = LayoutTree::new();
        assert!(!tree.reuse(LayoutNodeId(9999)));
        assert_eq!(
            tree.set_style(LayoutNodeId(9999), LayoutStyle::default()),
            Err(LayoutError::UnknownNode(LayoutNodeId(9999)))
        );
        assert!(tree.layout_of(LayoutNodeId(9999)).is_err());
    }

    #[test]
    fn a_three_level_tree_lays_out_and_reuses_wholesale() -> Result<(), LayoutError> {
        let mut tree = LayoutTree::new();
        tree.begin_frame();
        let leaf = tree.request_layout(sized(20.0, 30.0), &[])?;
        let middle = tree.request_layout(LayoutStyle::default(), &[leaf])?;
        let root = tree.request_layout(sized(100.0, 100.0), &[middle])?;
        tree.compute_layout(root, definite(100.0, 100.0))?;
        assert_eq!(tree.stats().nodes_created, 3);
        tree.end_frame();

        let leaf_rect = tree.layout_of(leaf)?;
        assert_eq!(leaf_rect.width, 20.0);
        assert_eq!(leaf_rect.height, 30.0);

        tree.begin_frame();
        for node in [leaf, middle, root] {
            assert!(tree.reuse(node));
        }
        assert_eq!(tree.stats().nodes_created, 0);
        assert_eq!(tree.stats().nodes_reused, 3);
        assert_eq!(tree.end_frame(), 0);
        assert_eq!(tree.live_node_count(), 3);
        Ok(())
    }

    #[test]
    fn children_can_be_relinked_without_recreating_the_parent() -> Result<(), LayoutError> {
        let mut tree = LayoutTree::new();
        tree.begin_frame();
        let first = tree.request_layout(sized(10.0, 10.0), &[])?;
        let parent = tree.request_layout(LayoutStyle::default(), &[first])?;
        tree.end_frame();

        tree.begin_frame();
        assert!(tree.reuse(parent));
        let second = tree.request_layout(sized(10.0, 10.0), &[])?;
        tree.set_children(parent, &[second])?;
        assert_eq!(tree.children(parent)?, vec![second]);
        // `first` went untouched, so the sweep reclaims it.
        assert_eq!(tree.end_frame(), 1);
        assert!(!tree.is_live(first));
        assert!(tree.is_live(parent));
        Ok(())
    }

    #[test]
    fn an_empty_tree_sweeps_nothing() {
        let mut tree = LayoutTree::new();
        tree.begin_frame();
        assert_eq!(tree.end_frame(), 0);
        assert_eq!(tree.live_node_count(), 0);
    }
}
