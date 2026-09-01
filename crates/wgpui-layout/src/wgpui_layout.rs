//! `wgpui-layout` — Taffy integration, isolated. See
//! docs/gpu-native-architecture.md §3.2. Depended on for heterogeneous
//! flexbox/grid layout, which stays on the CPU on purpose (§6); the regular-
//! content GPU layout kernel (§6.1) lives in `wgpui-core::shaders` /
//! `wgpui-wgpu::render::compute::layout_pass`, not here.
//!
//! [`taffy_tree`] provides the retained exact layout engine, while
//! [`containment`] and [`measure`] provide the optional, backend-neutral
//! intrinsic-size contract used to seed unresolved dimensions. Estimates are
//! strictly opt-in and invalid values fall back to exact Taffy layout.

pub mod containment;
pub mod measure;
pub mod regular;
pub mod taffy_tree;

pub use containment::{EstimatedSize, resolve_element_style, resolve_estimated_style};
pub use measure::{IntrinsicSize, LayoutSize as MeasureSize, Measure};

pub use taffy_tree::{
    AvailableSpace, Dimension, Display, FlexDirection, LayoutError, LayoutFrameStats, LayoutNodeId,
    LayoutRect, LayoutSize, LayoutStyle, LayoutTree, definite,
};

impl LayoutTree {
    /// Create a retained node using an optional intrinsic estimate.
    ///
    /// `None` and invalid estimates call the same exact `request_layout` path
    /// as callers that do not participate in containment. Children are kept in
    /// the tree, so flex, grid, min/max, padding, borders, and Taffy's rounding
    /// semantics remain authoritative.
    pub fn request_layout_with_estimate(
        &mut self,
        style: LayoutStyle,
        children: &[LayoutNodeId],
        estimate: Option<IntrinsicSize>,
    ) -> Result<LayoutNodeId, LayoutError> {
        self.request_layout(resolve_estimated_style(style, estimate), children)
    }

    /// Create a retained node using an element's optional intrinsic estimate.
    pub fn request_layout_for<E: EstimatedSize + ?Sized>(
        &mut self,
        style: LayoutStyle,
        children: &[LayoutNodeId],
        element: &E,
    ) -> Result<LayoutNodeId, LayoutError> {
        self.request_layout_with_estimate(style, children, element.estimated_size())
    }

    /// Update a retained node using an optional intrinsic estimate.
    ///
    /// Taffy receives the resolved style through its normal `set_style` call,
    /// so changing an effective estimate participates in ordinary dirty
    /// propagation. If the estimate does not affect an auto dimension, the
    /// authored style remains unchanged.
    pub fn set_style_with_estimate(
        &mut self,
        node: LayoutNodeId,
        style: LayoutStyle,
        estimate: Option<IntrinsicSize>,
    ) -> Result<(), LayoutError> {
        self.set_style(node, resolve_estimated_style(style, estimate))
    }

    /// Update a retained node using an element's optional intrinsic estimate.
    pub fn set_style_for<E: EstimatedSize + ?Sized>(
        &mut self,
        node: LayoutNodeId,
        style: LayoutStyle,
        element: &E,
    ) -> Result<(), LayoutError> {
        self.set_style_with_estimate(node, style, element.estimated_size())
    }
}

#[cfg(test)]
mod estimated_layout_tests {
    use super::*;
    use crate::taffy_tree::{Dimension, Display, LayoutRect, TaffySize};

    struct TestElement(Option<IntrinsicSize>);

    impl EstimatedSize for TestElement {
        fn estimated_size(&self) -> Option<IntrinsicSize> {
            self.0
        }
    }

    fn auto_style() -> LayoutStyle {
        LayoutStyle::default()
    }

    fn fixed_style(width: f32, height: f32) -> LayoutStyle {
        LayoutStyle {
            size: TaffySize {
                width: Dimension::length(width),
                height: Dimension::length(height),
            },
            ..LayoutStyle::default()
        }
    }

    fn leaf_tree(style: LayoutStyle) -> Result<(LayoutTree, LayoutNodeId), LayoutError> {
        let mut tree = LayoutTree::new();
        let node = tree.request_layout(style, &[])?;
        tree.compute_layout(node, definite(200.0, 200.0))?;
        Ok((tree, node))
    }

    #[test]
    fn a_valid_estimate_matches_the_exact_layout_for_matching_intrinsic_content()
    -> Result<(), LayoutError> {
        let (exact_tree, exact_node) = leaf_tree(fixed_style(48.0, 24.0))?;
        let mut estimated_tree = LayoutTree::new();
        let estimated_node = estimated_tree.request_layout_with_estimate(
            auto_style(),
            &[],
            Some(IntrinsicSize::new(48.0, 24.0)),
        )?;
        estimated_tree.compute_layout(estimated_node, definite(200.0, 200.0))?;

        assert_eq!(
            estimated_tree.layout_of(estimated_node)?,
            exact_tree.layout_of(exact_node)?
        );
        Ok(())
    }

    #[test]
    fn an_element_provider_uses_the_same_optional_estimate_contract() {
        let element = TestElement(Some(IntrinsicSize::new(32.0, 16.0)));
        let resolved = resolve_element_style(LayoutStyle::default(), &element);

        assert_eq!(resolved.size.width, Dimension::length(32.0));
        assert_eq!(resolved.size.height, Dimension::length(16.0));
    }

    #[test]
    fn estimates_preserve_nested_taffy_layout() -> Result<(), LayoutError> {
        let mut exact_tree = LayoutTree::new();
        let exact_child = exact_tree.request_layout(fixed_style(40.0, 18.0), &[])?;
        let exact_parent = exact_tree.request_layout(auto_style(), &[exact_child])?;
        let exact_root = exact_tree.request_layout(fixed_style(160.0, 100.0), &[exact_parent])?;
        exact_tree.compute_layout(exact_root, definite(160.0, 100.0))?;

        let mut estimated_tree = LayoutTree::new();
        let estimated_child = estimated_tree.request_layout(fixed_style(40.0, 18.0), &[])?;
        let estimated_parent = estimated_tree.request_layout_with_estimate(
            auto_style(),
            &[estimated_child],
            Some(IntrinsicSize::new(40.0, 100.0)),
        )?;
        let estimated_root =
            estimated_tree.request_layout(fixed_style(160.0, 100.0), &[estimated_parent])?;
        estimated_tree.compute_layout(estimated_root, definite(160.0, 100.0))?;

        for (exact, estimated) in [
            (exact_child, estimated_child),
            (exact_parent, estimated_parent),
            (exact_root, estimated_root),
        ] {
            assert_eq!(
                estimated_tree.layout_of(estimated)?,
                exact_tree.layout_of(exact)?
            );
        }
        Ok(())
    }

    #[test]
    fn missing_and_invalid_estimates_use_the_exact_path() -> Result<(), LayoutError> {
        let style = LayoutStyle {
            display: Display::Flex,
            size: TaffySize {
                width: Dimension::auto(),
                height: Dimension::auto(),
            },
            ..LayoutStyle::default()
        };
        let (exact_tree, exact_node) = leaf_tree(style.clone())?;

        for estimate in [
            None,
            Some(IntrinsicSize::new(-1.0, 10.0)),
            Some(IntrinsicSize::new(10.0, f32::INFINITY)),
        ] {
            let (tree, node) = leaf_tree(resolve_estimated_style(style.clone(), estimate))?;
            assert_eq!(tree.layout_of(node)?, exact_tree.layout_of(exact_node)?);
        }
        Ok(())
    }

    #[test]
    fn changing_an_estimate_invalidates_the_retained_layout() -> Result<(), LayoutError> {
        let style = auto_style();
        let mut tree = LayoutTree::new();
        let node = tree.request_layout_with_estimate(
            style.clone(),
            &[],
            Some(IntrinsicSize::new(20.0, 12.0)),
        )?;
        tree.compute_layout(node, definite(200.0, 200.0))?;
        assert_eq!(tree.layout_of(node)?.width, 20.0);

        tree.set_style_with_estimate(node, style, Some(IntrinsicSize::new(72.0, 36.0)))?;
        tree.compute_layout(node, definite(200.0, 200.0))?;

        assert_eq!(
            tree.layout_of(node)?,
            LayoutRect {
                x: 0.0,
                y: 0.0,
                width: 72.0,
                height: 36.0,
            }
        );
        Ok(())
    }
}
