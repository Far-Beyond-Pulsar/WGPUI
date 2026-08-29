//! The production painter-order algorithm, ported: a dynamic AABB tree that
//! answers "the highest order among already-inserted bounds intersecting this
//! one" without visiting every one of them.
//! See docs/gpu-native-architecture.md §5.1; ported from `src/bounds_tree.rs`.
//!
//! # Why a port lives here at all
//!
//! Two reasons, and neither is "so `wgpui-core` has a tree."
//!
//! 1. **It is the CPU path Phase 3's gate measures against.** §8's Phase 3 row
//!    asks for Phase 0's spike numbers reproduced "on the real pipeline"; the
//!    thing the compute pass has to beat is what the renderer does today, which
//!    is this algorithm, not the `O(n²)` definition in the parent module.
//! 2. **It is the second opinion the definition is checked against.** The tree
//!    and [`super::painter_orders`] must agree on every input, and the tests
//!    below assert that on randomised scenes — the same technique
//!    `bounds_tree.rs`'s own `test_random_iterations` uses, kept because it is
//!    what catches a pruning bug that a hand-written case would not.
//!
//! # What the port drops, and why that is safe
//!
//! `order_floor`, `insert_above_all`, and `insert_at_order` are not here. All
//! three exist for content-filter group boundaries and deferred-draw overlays
//! (`src/scene.rs`'s layer scopes), which are scene-assembly concerns that
//! `wgpui-core`'s layer model expresses as separate layers rather than as order
//! floors inside one. Adding them back is mechanical if a later phase needs
//! them; carrying them now would be three untested branches.
//!
//! The recursion in `find_max_ordering` is an explicit stack here. A layer of a
//! hundred thousand primitives can build a deep tree, and a recursive descent
//! over it is a stack-overflow risk in a library. The result is unaffected —
//! the traversal computes a maximum, which does not depend on visit order — and
//! the higher-`max_order` child is still visited first, which is what makes the
//! pruning effective.

use crate::geometry::Rect;

#[derive(Clone, Debug)]
enum Node {
    Leaf {
        bounds: Rect,
        order: u32,
    },
    Internal {
        left: usize,
        right: usize,
        bounds: Rect,
        max_order: u32,
    },
}

impl Node {
    fn bounds(&self) -> Rect {
        match self {
            Node::Leaf { bounds, .. } => *bounds,
            Node::Internal { bounds, .. } => *bounds,
        }
    }

    fn max_order(&self) -> u32 {
        match self {
            Node::Leaf { order, .. } => *order,
            Node::Internal { max_order, .. } => *max_order,
        }
    }
}

/// A dynamic AABB tree over one layer's primitives, in paint order.
#[derive(Clone, Debug, Default)]
pub struct BoundsTree {
    root: Option<usize>,
    nodes: Vec<Node>,
    /// The chain of internal nodes descended through on the current insert,
    /// walked back up afterwards to propagate `max_order`.
    parents: Vec<usize>,
    /// Traversal scratch for [`Self::find_max_order`], kept as a field so a
    /// hot insert loop does not reallocate it per call.
    search: Vec<usize>,
}

impl BoundsTree {
    /// An empty tree.
    pub fn new() -> Self {
        Self::default()
    }

    /// Forget every insert, keeping the allocations.
    pub fn clear(&mut self) {
        self.root = None;
        self.nodes.clear();
        self.parents.clear();
        self.search.clear();
    }

    /// The highest order assigned so far, or zero for an empty tree.
    pub fn max_order(&self) -> u32 {
        match self.root.and_then(|index| self.nodes.get(index)) {
            Some(node) => node.max_order(),
            None => 0,
        }
    }

    /// Insert `bounds` and return its painter order:
    /// `1 + max(order of every intersecting bounds already inserted)`.
    pub fn insert(&mut self, bounds: Rect) -> u32 {
        let Some(mut index) = self.root else {
            let leaf = self.push_leaf(bounds, 1);
            self.root = Some(leaf);
            return 1;
        };

        let mut max_intersecting = 0u32;
        // Descend to the best-fit leaf, collecting the highest intersecting
        // order out of every subtree not descended into.
        // The descent stops at a leaf or — unreachable by construction — a
        // dangling index; either way the pattern stops matching.
        while let Some(Node::Internal {
            left,
            right,
            bounds: node_bounds,
            ..
        }) = self.nodes.get_mut(index)
        {
            *node_bounds = node_bounds.union(&bounds);
            let (left, right) = (*left, *right);
            self.parents.push(index);

            let left_cost = self.child_cost(left, bounds);
            let right_cost = self.child_cost(right, bounds);
            if left_cost < right_cost {
                max_intersecting = self.find_max_order(right, bounds, max_intersecting);
                index = left;
            } else {
                max_intersecting = self.find_max_order(left, bounds, max_intersecting);
                index = right;
            }
        }

        let sibling = index;
        if let Some(Node::Leaf {
            bounds: sibling_bounds,
            order: sibling_order,
        }) = self.nodes.get(sibling)
            && sibling_bounds.intersects(&bounds)
            && *sibling_order > max_intersecting
        {
            max_intersecting = *sibling_order;
        }

        let order = max_intersecting + 1;
        self.attach_leaf(sibling, bounds, order);
        order
    }

    /// The surface-area-heuristic cost of descending into `child`.
    fn child_cost(&self, child: usize, bounds: Rect) -> f32 {
        match self.nodes.get(child) {
            // An absent child is unreachable by construction; costing it at
            // infinity makes the descent prefer the sibling rather than follow
            // a dangling index.
            None => f32::INFINITY,
            Some(node) => bounds.union(&node.bounds()).half_perimeter(),
        }
    }

    fn attach_leaf(&mut self, sibling: usize, bounds: Rect, order: u32) {
        let leaf = self.push_leaf(bounds, order);
        let parent = self.push_internal(sibling, leaf);

        match self.parents.last().copied() {
            Some(old_parent) => {
                if let Some(Node::Internal { left, right, .. }) = self.nodes.get_mut(old_parent) {
                    if *left == sibling {
                        *left = parent;
                    } else {
                        *right = parent;
                    }
                }
            }
            None => self.root = Some(parent),
        }

        while let Some(index) = self.parents.pop() {
            let Some(Node::Internal { max_order, .. }) = self.nodes.get_mut(index) else {
                continue;
            };
            if *max_order >= order {
                break;
            }
            *max_order = order;
        }
        self.parents.clear();
    }

    /// The highest order among leaves under `root` intersecting `bounds`,
    /// never below `best`.
    fn find_max_order(&mut self, root: usize, bounds: Rect, best: u32) -> u32 {
        let mut best = best;
        self.search.clear();
        self.search.push(root);
        while let Some(index) = self.search.pop() {
            match self.nodes.get(index) {
                Some(Node::Leaf {
                    bounds: node_bounds,
                    order,
                }) => {
                    if bounds.intersects(node_bounds) && *order > best {
                        best = *order;
                    }
                }
                Some(Node::Internal {
                    left,
                    right,
                    bounds: node_bounds,
                    max_order,
                }) => {
                    if !bounds.intersects(node_bounds) || best >= *max_order {
                        continue;
                    }
                    let left_max = self.nodes.get(*left).map_or(0, Node::max_order);
                    let right_max = self.nodes.get(*right).map_or(0, Node::max_order);
                    // Pushed lowest-max first so the highest-max child is
                    // popped first, raising `best` sooner and pruning more.
                    if left_max > right_max {
                        self.search.push(*right);
                        self.search.push(*left);
                    } else {
                        self.search.push(*left);
                        self.search.push(*right);
                    }
                }
                None => {}
            }
        }
        best
    }

    fn push_leaf(&mut self, bounds: Rect, order: u32) -> usize {
        self.nodes.push(Node::Leaf { bounds, order });
        self.nodes.len() - 1
    }

    fn push_internal(&mut self, left: usize, right: usize) -> usize {
        let left_bounds = self.nodes.get(left).map_or(Rect::EMPTY, Node::bounds);
        let right_bounds = self.nodes.get(right).map_or(Rect::EMPTY, Node::bounds);
        let left_max = self.nodes.get(left).map_or(0, Node::max_order);
        let right_max = self.nodes.get(right).map_or(0, Node::max_order);
        self.nodes.push(Node::Internal {
            bounds: left_bounds.union(&right_bounds),
            left,
            right,
            max_order: left_max.max(right_max),
        });
        self.nodes.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ordering::painter_orders;

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Rect {
        Rect::from_origin_size([x, y], [width, height])
    }

    /// A small deterministic generator, so a failure is reproducible from the
    /// seed printed in the assertion rather than from a lucky rerun.
    struct Random(u64);

    impl Random {
        fn next_u32(&mut self) -> u32 {
            // SplitMix64, chosen because it is five lines and has no
            // dependency; the tests need spread, not cryptographic quality.
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut value = self.0;
            value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((value ^ (value >> 31)) >> 32) as u32
        }

        fn next_in(&mut self, range: u32) -> u32 {
            if range == 0 {
                0
            } else {
                self.next_u32() % range
            }
        }
    }

    fn random_scene(seed: u64, count: usize, spread: u32, size: u32) -> Vec<Rect> {
        let mut random = Random(seed);
        (0..count)
            .map(|_| {
                let x = random.next_in(spread) as f32;
                let y = random.next_in(spread) as f32;
                let width = (random.next_in(size) + 1) as f32;
                let height = (random.next_in(size) + 1) as f32;
                rect(x, y, width, height)
            })
            .collect()
    }

    #[test]
    fn a_single_insert_is_order_one() {
        let mut tree = BoundsTree::new();
        assert_eq!(tree.insert(rect(0.0, 0.0, 10.0, 10.0)), 1);
        assert_eq!(tree.max_order(), 1);
    }

    #[test]
    fn disjoint_inserts_stay_at_order_one() {
        let mut tree = BoundsTree::new();
        assert_eq!(tree.insert(rect(0.0, 0.0, 10.0, 10.0)), 1);
        assert_eq!(tree.insert(rect(50.0, 0.0, 10.0, 10.0)), 1);
        assert_eq!(tree.insert(rect(100.0, 0.0, 10.0, 10.0)), 1);
    }

    #[test]
    fn overlapping_inserts_step_by_one() {
        let mut tree = BoundsTree::new();
        assert_eq!(tree.insert(rect(0.0, 0.0, 10.0, 10.0)), 1);
        assert_eq!(tree.insert(rect(5.0, 5.0, 10.0, 10.0)), 2);
        assert_eq!(tree.insert(rect(6.0, 6.0, 10.0, 10.0)), 3);
        assert_eq!(tree.max_order(), 3);
    }

    #[test]
    fn clearing_resets_the_tree() {
        let mut tree = BoundsTree::new();
        tree.insert(rect(0.0, 0.0, 10.0, 10.0));
        tree.insert(rect(1.0, 1.0, 10.0, 10.0));
        tree.clear();
        assert_eq!(tree.max_order(), 0);
        assert_eq!(tree.insert(rect(1.0, 1.0, 10.0, 10.0)), 1);
    }

    #[test]
    fn the_tree_agrees_with_the_definition_on_dense_random_scenes() {
        for seed in 0..8u64 {
            let bounds = random_scene(seed, 200, 100, 30);
            assert_eq!(
                super::super::painter_orders_via_tree(&bounds),
                painter_orders(&bounds),
                "tree and recurrence disagree on dense seed {seed}"
            );
        }
    }

    #[test]
    fn the_tree_agrees_with_the_definition_on_sparse_random_scenes() {
        for seed in 100..108u64 {
            let bounds = random_scene(seed, 300, 5_000, 20);
            assert_eq!(
                super::super::painter_orders_via_tree(&bounds),
                painter_orders(&bounds),
                "tree and recurrence disagree on sparse seed {seed}"
            );
        }
    }

    #[test]
    fn the_tree_agrees_with_the_definition_when_everything_overlaps() {
        // Every primitive covers the origin, so the orders are 1, 2, 3, ... and
        // the pruning never gets to fire — the worst case for the tree, and the
        // one most likely to expose an off-by-one in `max_order` propagation.
        let bounds: Vec<Rect> = (0..64)
            .map(|index| rect(0.0, 0.0, 100.0 + index as f32, 100.0))
            .collect();
        let orders = super::super::painter_orders_via_tree(&bounds);
        assert_eq!(orders, painter_orders(&bounds));
        assert_eq!(orders.last().copied(), Some(64));
    }

    #[test]
    fn degenerate_bounds_never_intersect_anything() {
        let bounds = [
            rect(0.0, 0.0, 10.0, 10.0),
            rect(0.0, 0.0, 0.0, 10.0),
            rect(0.0, 0.0, 10.0, 0.0),
        ];
        assert_eq!(
            super::super::painter_orders_via_tree(&bounds),
            vec![1, 1, 1]
        );
        assert_eq!(painter_orders(&bounds), vec![1, 1, 1]);
    }
}
