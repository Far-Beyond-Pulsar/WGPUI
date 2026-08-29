//! Painter ordering within one layer: what `BoundsTree` computes today, stated
//! as a recurrence so it can be solved data-parallel.
//! See docs/gpu-native-architecture.md §5.1, R-N §5.1.
//!
//! Not in §3's file map — §3.1 gives `shaders/ordering.wgsl` a home and gives
//! the computation it implements none, because in the legacy backend that
//! computation *is* `src/bounds_tree.rs` sitting next to `scene.rs`. Phase 3
//! needs three things in one place instead: the definition, a faithful port of
//! the production algorithm to check it against, and the GPU-side encoding.
//! That is one module's worth of concern, and putting it in `scene.rs` would
//! have made that file the thing §3 exists to prevent (the same reasoning
//! `scene/primitive_store.rs` records for its own existence).
//!
//! # The recurrence
//!
//! `BoundsTree::insert` (`src/bounds_tree.rs:61`) assigns each newly inserted
//! bounds `max(order of every already-inserted bounds it intersects) + 1`. Over
//! a layer's primitives in paint order that is exactly:
//!
//! ```text
//! order[i] = 1 + max{ order[j] : j < i and bounds[j] intersects bounds[i] }
//! ```
//!
//! with the max over an empty set taken as zero. The AABB tree is a way to
//! answer the inner query quickly, not a different answer — [`bounds_tree`]'s
//! own tests assert the two agree on random scenes, which is the same technique
//! `bounds_tree.rs`'s `test_random_iterations` uses in the legacy backend.
//!
//! **Why that matters for the compute path.** The recurrence is monotone and
//! references only strictly earlier indices, so iterating it from an all-ones
//! initial state converges to the unique fixed point, and after `k` iterations
//! every primitive whose true order is at most `k + 1` already holds it. That
//! is what makes Jacobi relaxation over ping-ponged buffers (Phase 0's Spike A,
//! validated bit-for-bit there) a *solution* rather than an approximation: the
//! compute pass runs the iteration to a fixed point and checks that it reached
//! one, rather than running a fixed budget and hoping.
//!
//! # What this module does not decide
//!
//! `order` is not a slot address. Phase 2 established that a record keeps the
//! slot it was inserted at, and Phase 3 does not move bytes: the ordering pass
//! produces a *draw permutation* over the layer's existing residency. What
//! consumes that permutation is Phase 4's indirect draw-arg generation (§5.3);
//! this phase feeds it into the same CPU-side draw-range decision
//! [`crate::scene::Scene::draw_ranges`] already makes.

pub mod bounds_tree;

use crate::geometry::Rect;

/// How many primitives one leaf block of the compute pass's AABB hierarchy
/// covers. Also the ordering shader's workgroup size.
pub const BLOCK_SIZE: u32 = 64;

/// How many blocks one superblock covers, so a superblock spans
/// `BLOCK_SIZE * SUPERBLOCK_SIZE` primitives.
pub const SUPERBLOCK_SIZE: u32 = 64;

/// Bytes one primitive's ordering input occupies: a `vec4<f32>` of bounds.
pub const ORDERING_ITEM_STRIDE: usize = 16;

/// Bytes one hierarchy node occupies: a `vec4<f32>` of bounds.
pub const ORDERING_NODE_STRIDE: usize = 16;

/// The recurrence itself, evaluated directly. `O(n²)`.
///
/// This is the *definition* of a painter order, and it is what the compute path
/// and [`bounds_tree`] are both checked against. It is not a path any real
/// frame takes — a layer of ten thousand primitives is a hundred million pair
/// tests here — and exists so "the tree and the shader compute the same thing"
/// is a claim with something to be true *of*.
pub fn painter_orders(bounds: &[Rect]) -> Vec<u32> {
    let mut orders = vec![0u32; bounds.len()];
    for index in 0..bounds.len() {
        let Some(current) = bounds.get(index) else {
            continue;
        };
        let mut best = 0u32;
        for probe in 0..index {
            let (Some(earlier), Some(order)) = (bounds.get(probe), orders.get(probe)) else {
                continue;
            };
            if earlier.intersects(current) && *order > best {
                best = *order;
            }
        }
        orders[index] = best + 1;
    }
    orders
}

/// The same recurrence, answered through the production AABB tree — the CPU
/// path Phase 3's compute pass is measured against.
pub fn painter_orders_via_tree(bounds: &[Rect]) -> Vec<u32> {
    let mut tree = bounds_tree::BoundsTree::new();
    bounds.iter().map(|item| tree.insert(*item)).collect()
}

/// Indices sorted into draw order: ascending `order`, ties broken by original
/// index so the result is a stable painter's order.
///
/// The tie-break is not cosmetic. `Scene::finish` sorts with `sort_by_key`,
/// which is stable, so two non-overlapping primitives that share an order keep
/// their emission order; a bitonic sort is not stable, so the compute path has
/// to carry the index into the comparison to reproduce this. Both sides
/// therefore compare `(order, index)` rather than `order` alone.
pub fn draw_order(orders: &[u32]) -> Vec<u32> {
    let mut indices: Vec<u32> = (0..orders.len())
        .map(|index| u32::try_from(index).unwrap_or(u32::MAX))
        .collect();
    indices.sort_by_key(|index| {
        let position = usize::try_from(*index).unwrap_or(usize::MAX);
        (orders.get(position).copied().unwrap_or(0), *index)
    });
    indices
}

/// Encode a layer's bounds for `shaders/ordering.wgsl`.
///
/// Byte-oriented for the reason `patch/primitive.rs` gives: it keeps
/// `wgpui-core` dependency-free and makes the GPU layout explicit.
pub fn encode_ordering_items(bounds: &[Rect], destination: &mut Vec<u8>) {
    destination.clear();
    destination.reserve(bounds.len() * ORDERING_ITEM_STRIDE);
    for item in bounds {
        for value in item.to_array() {
            destination.extend_from_slice(&value.to_le_bytes());
        }
    }
}

/// How many leaf blocks a layer of `count` primitives needs.
pub fn block_count(count: u32) -> u32 {
    count.div_ceil(BLOCK_SIZE)
}

/// How many superblocks a layer of `count` primitives needs.
pub fn superblock_count(count: u32) -> u32 {
    block_count(count).div_ceil(SUPERBLOCK_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Rect {
        Rect::from_origin_size([x, y], [width, height])
    }

    #[test]
    fn disjoint_primitives_all_share_the_lowest_order() {
        let bounds = [
            rect(0.0, 0.0, 10.0, 10.0),
            rect(20.0, 0.0, 10.0, 10.0),
            rect(40.0, 0.0, 10.0, 10.0),
        ];
        assert_eq!(painter_orders(&bounds), vec![1, 1, 1]);
    }

    #[test]
    fn a_stack_of_overlapping_primitives_steps_by_one() {
        let bounds = [
            rect(0.0, 0.0, 10.0, 10.0),
            rect(1.0, 1.0, 10.0, 10.0),
            rect(2.0, 2.0, 10.0, 10.0),
        ];
        assert_eq!(painter_orders(&bounds), vec![1, 2, 3]);
    }

    #[test]
    fn order_never_looks_forward() {
        // The later primitive overlaps the earlier one, so only the later one
        // steps: painter order is a function of what is already painted.
        let bounds = [rect(0.0, 0.0, 100.0, 100.0), rect(0.0, 0.0, 1.0, 1.0)];
        assert_eq!(painter_orders(&bounds), vec![1, 2]);
    }

    #[test]
    fn touching_edges_do_not_step_the_order() {
        let bounds = [rect(0.0, 0.0, 10.0, 10.0), rect(10.0, 0.0, 10.0, 10.0)];
        assert_eq!(painter_orders(&bounds), vec![1, 1]);
    }

    #[test]
    fn an_empty_layer_orders_nothing() {
        assert!(painter_orders(&[]).is_empty());
        assert!(draw_order(&[]).is_empty());
    }

    #[test]
    fn draw_order_is_stable_across_equal_orders() {
        let orders = [2, 1, 2, 1, 3];
        assert_eq!(draw_order(&orders), vec![1, 3, 0, 2, 4]);
    }

    #[test]
    fn encoding_produces_one_vec4_per_primitive() {
        let bounds = [rect(1.0, 2.0, 3.0, 4.0)];
        let mut bytes = Vec::new();
        encode_ordering_items(&bounds, &mut bytes);
        assert_eq!(bytes.len(), ORDERING_ITEM_STRIDE);
        assert_eq!(&bytes[0..4], &1.0f32.to_le_bytes());
        assert_eq!(&bytes[8..12], &4.0f32.to_le_bytes());
    }

    #[test]
    fn hierarchy_sizing_covers_every_primitive() {
        assert_eq!(block_count(0), 0);
        assert_eq!(block_count(1), 1);
        assert_eq!(block_count(BLOCK_SIZE), 1);
        assert_eq!(block_count(BLOCK_SIZE + 1), 2);
        assert_eq!(superblock_count(BLOCK_SIZE * SUPERBLOCK_SIZE), 1);
        assert_eq!(superblock_count(BLOCK_SIZE * SUPERBLOCK_SIZE + 1), 2);
    }
}
