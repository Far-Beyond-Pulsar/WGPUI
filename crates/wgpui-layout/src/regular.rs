//! The backend-neutral contract for regular-content layout.
//!
//! A regular line is a single, non-wrapping flex line whose children have
//! known sizes. Flex sizing is resolved while lowering the retained styles;
//! item placement is then independent and can be dispatched one item per
//! invocation. The GPU path consumes the packed representation, while
//! [`RegularLayoutInput::compute_cpu`] is the exact reference used for
//! fallback and differential tests.

use crate::taffy_tree::{
    AlignContent, AlignItems, BoxSizing, Dimension, Display, FlexDirection, FlexWrap,
    JustifyContent, LayoutRect, LayoutStyle, LengthPercentage, LengthPercentageAuto, Position,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegularLayoutItem {
    pub size: [f32; 2],
    pub min_size: [f32; 2],
    pub max_size: [f32; 2],
    pub flex: [f32; 2],
    pub margin: [f32; 4],
    /// Axis-aligned affine transform `[a, b, c, d, tx, ty]`.
    pub transform: [f32; 6],
}

impl RegularLayoutItem {
    pub fn fixed(width: f32, height: f32) -> Self {
        Self {
            size: [width, height],
            min_size: [0.0; 2],
            max_size: [f32::INFINITY; 2],
            flex: [0.0, 1.0],
            margin: [0.0; 4],
            transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum RegularJustifyContent {
    #[default]
    Start = 0,
    End = 1,
    Center = 2,
    SpaceBetween = 3,
    SpaceAround = 4,
    SpaceEvenly = 5,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum RegularAlignItems {
    Start = 0,
    End = 1,
    Center = 2,
    #[default]
    Stretch = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum RegularAxis {
    Row = 0,
    Column = 1,
    RowReverse = 2,
    ColumnReverse = 3,
}

impl RegularAxis {
    fn is_row(self) -> bool {
        matches!(self, Self::Row | Self::RowReverse)
    }
    fn is_reverse(self) -> bool {
        matches!(self, Self::RowReverse | Self::ColumnReverse)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegularTransform(pub [f32; 6]);

impl Default for RegularTransform {
    fn default() -> Self {
        Self([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegularLayoutInput {
    pub origin: [f32; 2],
    pub container_size: [f32; 2],
    pub padding: [f32; 4],
    pub gap: f32,
    pub axis: RegularAxis,
    pub justify_content: RegularJustifyContent,
    pub align_items: RegularAlignItems,
    pub rounding_scale: f32,
    pub items: Vec<RegularLayoutItem>,
}

impl RegularLayoutInput {
    pub fn validate(&self) -> Result<(), RegularLayoutFallback> {
        if self.items.is_empty() {
            return Err(RegularLayoutFallback::EmptyContent);
        }
        if self.items.len() > u32::MAX as usize
            || self.items.len() > usize::MAX / REGULAR_LAYOUT_ITEM_STRIDE
        {
            return Err(RegularLayoutFallback::InvalidNumber);
        }
        if !self
            .origin
            .iter()
            .chain(self.container_size.iter())
            .all(|v| v.is_finite())
            || !self.padding.iter().all(|v| v.is_finite() && *v >= 0.0)
            || !self.gap.is_finite()
            || self.gap < 0.0
            || !self.rounding_scale.is_finite()
            || self.rounding_scale <= 0.0
        {
            return Err(RegularLayoutFallback::InvalidNumber);
        }
        if self.container_size.iter().any(|v| *v < 0.0) {
            return Err(RegularLayoutFallback::InvalidContainerSize);
        }
        for item in &self.items {
            if !item.size.iter().all(|v| v.is_finite() && *v >= 0.0)
                || !item.min_size.iter().all(|v| v.is_finite() && *v >= 0.0)
                || !item
                    .max_size
                    .iter()
                    .all(|v| (*v >= 0.0 && v.is_finite()) || *v == f32::INFINITY)
                || !item.flex.iter().all(|v| v.is_finite() && *v >= 0.0)
                || !item.margin.iter().all(|v| v.is_finite() && *v >= 0.0)
                || !item.transform.iter().all(|v| v.is_finite())
            {
                return Err(RegularLayoutFallback::InvalidNumber);
            }
            if item
                .min_size
                .iter()
                .zip(item.max_size.iter())
                .any(|(min, max)| min > max)
            {
                return Err(RegularLayoutFallback::InvalidConstraint);
            }
            let [a, b, c, d, _, _] = item.transform;
            if b.abs() > f32::EPSILON || c.abs() > f32::EPSILON || a < 0.0 || d < 0.0 {
                return Err(RegularLayoutFallback::UnsupportedTransform);
            }
        }
        Ok(())
    }

    pub fn compute_cpu(&self) -> Result<Vec<LayoutRect>, RegularLayoutFallback> {
        self.validate()?;
        let resolved_sizes = self.resolve_main_sizes()?;
        let row = self.axis.is_row();
        let main_size = if row {
            self.container_size[0]
        } else {
            self.container_size[1]
        };
        let cross_size = if row {
            self.container_size[1]
        } else {
            self.container_size[0]
        };
        let main_start = if row {
            self.padding[0]
        } else {
            self.padding[1]
        };
        let main_end = if row {
            self.padding[2]
        } else {
            self.padding[3]
        };
        let cross_start = if row {
            self.padding[1]
        } else {
            self.padding[0]
        };
        let cross_end = if row {
            self.padding[3]
        } else {
            self.padding[2]
        };
        let available_main = (main_size - main_start - main_end).max(0.0);
        let available_cross = (cross_size - cross_start - cross_end).max(0.0);
        let occupied: f32 = resolved_sizes
            .iter()
            .zip(&self.items)
            .map(|(size, item)| {
                main(*size, row)
                    + if row {
                        item.margin[0] + item.margin[2]
                    } else {
                        item.margin[1] + item.margin[3]
                    }
            })
            .sum::<f32>()
            + self.gap * self.items.len().saturating_sub(1) as f32;
        let free = available_main - occupied;
        let (leading, extra_gap) = justify(self.justify_content, free, self.items.len());
        let mut cursor = main_start + leading;
        let mut output = Vec::with_capacity(self.items.len());
        for (size, item) in resolved_sizes.iter().zip(&self.items) {
            let main_length = main(*size, row);
            let cross_length = cross(*size, row).min(available_cross);
            let cross_margin_start = if row { item.margin[1] } else { item.margin[0] };
            let cross_margin_end = if row { item.margin[3] } else { item.margin[2] };
            let cross_free = available_cross - cross_length - cross_margin_start - cross_margin_end;
            let cross_offset = match self.align_items {
                RegularAlignItems::Start | RegularAlignItems::Stretch => cross_margin_start,
                RegularAlignItems::End => cross_margin_start + cross_free.max(0.0),
                RegularAlignItems::Center => cross_margin_start + cross_free.max(0.0) / 2.0,
            };
            cursor += if row { item.margin[0] } else { item.margin[1] };
            let main_position = if self.axis.is_reverse() {
                available_main + main_start - (cursor - main_start) - main_length
            } else {
                cursor
            };
            let cross_position = cross_start + cross_offset;
            let (x, y, width, height) = if row {
                (main_position, cross_position, main_length, cross_length)
            } else {
                (cross_position, main_position, cross_length, main_length)
            };
            output.push(transform_and_round(
                self.origin,
                LayoutRect {
                    x,
                    y,
                    width,
                    height,
                },
                item.transform,
                self.rounding_scale,
            ));
            cursor += main_length
                + if row { item.margin[2] } else { item.margin[3] }
                + self.gap
                + extra_gap;
        }
        Ok(output)
    }

    pub fn pack(&self) -> Result<RegularLayoutPacked, RegularLayoutFallback> {
        self.validate()?;
        let resolved_sizes = self.resolve_main_sizes()?;
        let mut params = Vec::with_capacity(REGULAR_LAYOUT_PARAMS_SIZE);
        for value in [
            self.items.len() as u32,
            self.axis as u32,
            self.justify_content as u32,
            self.align_items as u32,
        ] {
            push_u32(&mut params, value);
        }
        for value in [
            self.origin[0],
            self.origin[1],
            self.container_size[0],
            self.container_size[1],
        ] {
            push_f32(&mut params, value);
        }
        for value in self.padding {
            push_f32(&mut params, value);
        }
        for value in [self.gap, self.rounding_scale, 0.0, 0.0] {
            push_f32(&mut params, value);
        }
        let mut items = Vec::with_capacity(self.items.len() * REGULAR_LAYOUT_ITEM_STRIDE);
        for (resolved, item) in resolved_sizes.into_iter().zip(&self.items) {
            for value in [
                resolved[0],
                resolved[1],
                item.min_size[0],
                item.min_size[1],
                item.max_size[0],
                item.max_size[1],
                item.flex[0],
                item.flex[1],
            ] {
                push_f32(&mut items, value);
            }
            for value in item.margin {
                push_f32(&mut items, value);
            }
            for value in item.transform {
                push_f32(&mut items, value);
            }
            push_f32(&mut items, 0.0);
            push_f32(&mut items, 0.0);
        }
        Ok(RegularLayoutPacked { params, items })
    }

    fn resolve_main_sizes(&self) -> Result<Vec<[f32; 2]>, RegularLayoutFallback> {
        let row = self.axis.is_row();
        let main_size = if row {
            self.container_size[0]
        } else {
            self.container_size[1]
        };
        let content_main = (main_size
            - if row {
                self.padding[0] + self.padding[2]
            } else {
                self.padding[1] + self.padding[3]
            })
        .max(0.0);
        let gap_total = self.gap * self.items.len().saturating_sub(1) as f32;
        let mut sizes: Vec<[f32; 2]> = self
            .items
            .iter()
            .map(|item| {
                [
                    item.size[0].clamp(item.min_size[0], item.max_size[0]),
                    item.size[1].clamp(item.min_size[1], item.max_size[1]),
                ]
            })
            .collect();
        let occupied = sizes
            .iter()
            .zip(&self.items)
            .map(|(size, item)| {
                main(*size, row)
                    + if row {
                        item.margin[0] + item.margin[2]
                    } else {
                        item.margin[1] + item.margin[3]
                    }
            })
            .sum::<f32>()
            + gap_total;
        let free = content_main - occupied;
        if free > 0.0 {
            distribute(&mut sizes, &self.items, row, free, true);
        } else if free < 0.0 {
            distribute(&mut sizes, &self.items, row, free, false);
        }
        Ok(sizes)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegularLayoutPacked {
    pub params: Vec<u8>,
    pub items: Vec<u8>,
}

pub const REGULAR_LAYOUT_ITEM_STRIDE: usize = 80;
pub const REGULAR_LAYOUT_PARAMS_SIZE: usize = 64;
pub const DEFAULT_GPU_MIN_ITEMS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegularLayoutFallback {
    EmptyContent,
    InvalidNumber,
    InvalidContainerSize,
    InvalidConstraint,
    UnsupportedDisplay,
    UnsupportedWrap,
    UnsupportedPosition,
    UnsupportedBoxSizing,
    UnsupportedContentSize,
    UnsupportedAlignment,
    UnsupportedTransform,
    NestedContent,
    WorkloadTooSmall,
    DeviceUnsupported,
}

impl std::fmt::Display for RegularLayoutFallback {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::EmptyContent => "regular layout has no items",
            Self::InvalidNumber => "regular layout contains a non-finite or negative number",
            Self::InvalidContainerSize => "regular layout container size is negative",
            Self::InvalidConstraint => "regular layout min/max constraints are inconsistent",
            Self::UnsupportedDisplay => "regular layout requires flex display",
            Self::UnsupportedWrap => "regular layout requires a non-wrapping flex line",
            Self::UnsupportedPosition => {
                "regular layout does not support absolute or inset positioning"
            }
            Self::UnsupportedBoxSizing => "regular layout requires border-box sizing",
            Self::UnsupportedContentSize => "regular layout requires definite item dimensions",
            Self::UnsupportedAlignment => "regular layout cannot represent this alignment exactly",
            Self::UnsupportedTransform => {
                "regular layout requires an axis-aligned non-negative transform"
            }
            Self::NestedContent => "regular layout cannot lower nested content",
            Self::WorkloadTooSmall => "regular layout workload is below the GPU threshold",
            Self::DeviceUnsupported => "regular layout is unsupported by this device",
        })
    }
}
impl std::error::Error for RegularLayoutFallback {}

pub fn lower_styles(
    container: &LayoutStyle,
    child_styles: &[LayoutStyle],
    container_bounds: LayoutRect,
    rounding_scale: f32,
) -> Result<RegularLayoutInput, RegularLayoutFallback> {
    lower_styles_with_nesting(
        container,
        child_styles,
        container_bounds,
        rounding_scale,
        false,
    )
}

/// Like [`lower_styles`], with the retained walk's child-shape fact supplied
/// explicitly. A regular container cannot contain a nested layout because
/// that would make its child's size depend on another layout dispatch.
pub fn lower_styles_with_nesting(
    container: &LayoutStyle,
    child_styles: &[LayoutStyle],
    container_bounds: LayoutRect,
    rounding_scale: f32,
    nested_content: bool,
) -> Result<RegularLayoutInput, RegularLayoutFallback> {
    if nested_content {
        return Err(RegularLayoutFallback::NestedContent);
    }
    if container.display != Display::Flex {
        return Err(RegularLayoutFallback::UnsupportedDisplay);
    }
    if container.flex_wrap != FlexWrap::NoWrap {
        return Err(RegularLayoutFallback::UnsupportedWrap);
    }
    if container.position != Position::Relative
        || !container.inset.left.is_auto()
        || !container.inset.right.is_auto()
        || !container.inset.top.is_auto()
        || !container.inset.bottom.is_auto()
    {
        return Err(RegularLayoutFallback::UnsupportedPosition);
    }
    if container.box_sizing != BoxSizing::BorderBox {
        return Err(RegularLayoutFallback::UnsupportedBoxSizing);
    }
    if container.aspect_ratio.is_some()
        || matches!(
            container.align_content,
            Some(
                AlignContent::End
                    | AlignContent::FlexEnd
                    | AlignContent::Center
                    | AlignContent::SpaceBetween
                    | AlignContent::SpaceAround
                    | AlignContent::SpaceEvenly
            )
        )
    {
        return Err(RegularLayoutFallback::UnsupportedAlignment);
    }
    let axis = match container.flex_direction {
        FlexDirection::Row => RegularAxis::Row,
        FlexDirection::Column => RegularAxis::Column,
        FlexDirection::RowReverse => RegularAxis::RowReverse,
        FlexDirection::ColumnReverse => RegularAxis::ColumnReverse,
    };
    let padding = [
        resolve_length_percentage(container.padding.left, container_bounds.width)?
            + resolve_length_percentage(container.border.left, container_bounds.width)?,
        resolve_length_percentage(container.padding.top, container_bounds.height)?
            + resolve_length_percentage(container.border.top, container_bounds.height)?,
        resolve_length_percentage(container.padding.right, container_bounds.width)?
            + resolve_length_percentage(container.border.right, container_bounds.width)?,
        resolve_length_percentage(container.padding.bottom, container_bounds.height)?
            + resolve_length_percentage(container.border.bottom, container_bounds.height)?,
    ];
    let gap = if axis.is_row() {
        resolve_length_percentage(container.gap.width, container_bounds.width)?
    } else {
        resolve_length_percentage(container.gap.height, container_bounds.height)?
    };
    let mut items = Vec::with_capacity(child_styles.len());
    for style in child_styles {
        if style.display == Display::None {
            return Err(RegularLayoutFallback::UnsupportedDisplay);
        }
        if style.position != Position::Relative
            || !style.inset.left.is_auto()
            || !style.inset.right.is_auto()
            || !style.inset.top.is_auto()
            || !style.inset.bottom.is_auto()
        {
            return Err(RegularLayoutFallback::UnsupportedPosition);
        }
        if style.box_sizing != BoxSizing::BorderBox {
            return Err(RegularLayoutFallback::UnsupportedBoxSizing);
        }
        if style.aspect_ratio.is_some() {
            return Err(RegularLayoutFallback::UnsupportedContentSize);
        }
        let width = resolve_dimension(style.size.width, container_bounds.width)?;
        let height = resolve_dimension(style.size.height, container_bounds.height)?;
        let min_width = resolve_dimension_or_zero(style.min_size.width, container_bounds.width)?;
        let min_height = resolve_dimension_or_zero(style.min_size.height, container_bounds.height)?;
        let max_width =
            resolve_dimension_or_infinity(style.max_size.width, container_bounds.width)?;
        let max_height =
            resolve_dimension_or_infinity(style.max_size.height, container_bounds.height)?;
        let margin = [
            resolve_margin(style.margin.left, container_bounds.width)?,
            resolve_margin(style.margin.top, container_bounds.height)?,
            resolve_margin(style.margin.right, container_bounds.width)?,
            resolve_margin(style.margin.bottom, container_bounds.height)?,
        ];
        let mut size = [width, height];
        if let Some(basis) = resolve_dimension_or_none(
            style.flex_basis,
            if axis.is_row() {
                container_bounds.width
            } else {
                container_bounds.height
            },
        )? {
            if axis.is_row() {
                size[0] = basis;
            } else {
                size[1] = basis;
            }
        }
        items.push(RegularLayoutItem {
            size,
            min_size: [min_width, min_height],
            max_size: [max_width, max_height],
            flex: [style.flex_grow, style.flex_shrink],
            margin,
            transform: RegularTransform::default().0,
        });
    }
    let input = RegularLayoutInput {
        origin: [container_bounds.x, container_bounds.y],
        container_size: [container_bounds.width, container_bounds.height],
        padding,
        gap,
        axis,
        justify_content: map_justify(container.justify_content.unwrap_or(JustifyContent::Start)),
        align_items: map_align(container.align_items.unwrap_or(AlignItems::Stretch))?,
        rounding_scale,
        items,
    };
    input.validate()?;
    Ok(input)
}

fn resolve_dimension(value: Dimension, reference: f32) -> Result<f32, RegularLayoutFallback> {
    resolve_dimension_or_none(value, reference)?
        .ok_or(RegularLayoutFallback::UnsupportedContentSize)
}
fn resolve_dimension_or_zero(
    value: Dimension,
    reference: f32,
) -> Result<f32, RegularLayoutFallback> {
    Ok(resolve_dimension_or_none(value, reference)?.unwrap_or(0.0))
}
fn resolve_dimension_or_infinity(
    value: Dimension,
    reference: f32,
) -> Result<f32, RegularLayoutFallback> {
    Ok(resolve_dimension_or_none(value, reference)?.unwrap_or(f32::INFINITY))
}
fn resolve_dimension_or_none(
    value: Dimension,
    reference: f32,
) -> Result<Option<f32>, RegularLayoutFallback> {
    if value.is_auto() {
        return Ok(None);
    }
    let resolved = if value.tag() == Dimension::length(0.0).tag() {
        value.value()
    } else if value.tag() == Dimension::percent(0.0).tag() {
        reference * value.value()
    } else {
        return Err(RegularLayoutFallback::UnsupportedContentSize);
    };
    if resolved.is_finite() && resolved >= 0.0 {
        Ok(Some(resolved))
    } else {
        Err(RegularLayoutFallback::InvalidNumber)
    }
}
fn resolve_length_percentage(
    value: LengthPercentage,
    reference: f32,
) -> Result<f32, RegularLayoutFallback> {
    let raw = value.into_raw();
    let resolved = if raw.tag() == Dimension::length(0.0).into_raw().tag() {
        raw.value()
    } else if raw.tag() == Dimension::percent(0.0).into_raw().tag() {
        reference * raw.value()
    } else {
        return Err(RegularLayoutFallback::UnsupportedContentSize);
    };
    if resolved.is_finite() && resolved >= 0.0 {
        Ok(resolved)
    } else {
        Err(RegularLayoutFallback::InvalidNumber)
    }
}
fn resolve_margin(
    value: LengthPercentageAuto,
    reference: f32,
) -> Result<f32, RegularLayoutFallback> {
    value
        .resolve_to_option(reference, |_, _| 0.0)
        .filter(|number| number.is_finite() && *number >= 0.0)
        .ok_or(RegularLayoutFallback::UnsupportedContentSize)
}
fn map_justify(value: JustifyContent) -> RegularJustifyContent {
    match value {
        JustifyContent::Start | JustifyContent::FlexStart => RegularJustifyContent::Start,
        JustifyContent::End | JustifyContent::FlexEnd => RegularJustifyContent::End,
        JustifyContent::Center => RegularJustifyContent::Center,
        JustifyContent::SpaceBetween => RegularJustifyContent::SpaceBetween,
        JustifyContent::SpaceAround => RegularJustifyContent::SpaceAround,
        JustifyContent::SpaceEvenly => RegularJustifyContent::SpaceEvenly,
        JustifyContent::Stretch => RegularJustifyContent::Start,
    }
}
fn map_align(value: AlignItems) -> Result<RegularAlignItems, RegularLayoutFallback> {
    match value {
        AlignItems::Start | AlignItems::FlexStart => Ok(RegularAlignItems::Start),
        AlignItems::End | AlignItems::FlexEnd => Ok(RegularAlignItems::End),
        AlignItems::Center => Ok(RegularAlignItems::Center),
        AlignItems::Stretch => Ok(RegularAlignItems::Stretch),
        AlignItems::Baseline => Err(RegularLayoutFallback::UnsupportedAlignment),
    }
}
fn distribute(
    sizes: &mut [[f32; 2]],
    items: &[RegularLayoutItem],
    row: bool,
    free: f32,
    grow: bool,
) {
    let mut remaining = free;
    for _ in 0..sizes.len().saturating_add(1) {
        let weight: f32 = sizes
            .iter()
            .zip(items)
            .filter(|(size, item)| {
                let value = main(**size, row);
                let limit = if grow {
                    main(item.max_size, row)
                } else {
                    main(item.min_size, row)
                };
                (grow && value < limit) || (!grow && value > limit)
            })
            .map(|(size, item)| {
                if grow {
                    item.flex[0]
                } else {
                    item.flex[1] * main(*size, row)
                }
            })
            .sum();
        if weight <= 0.0 || remaining.abs() <= f32::EPSILON {
            break;
        }
        let mut consumed = 0.0;
        for (size, item) in sizes.iter_mut().zip(items) {
            let current = main(*size, row);
            let limit = if grow {
                main(item.max_size, row)
            } else {
                main(item.min_size, row)
            };
            let eligible = (grow && current < limit) || (!grow && current > limit);
            if !eligible {
                continue;
            }
            let item_weight = if grow {
                item.flex[0]
            } else {
                item.flex[1] * current
            };
            let delta = if grow {
                (remaining * item_weight / weight)
                    .min(limit - current)
                    .max(0.0)
            } else {
                (remaining * item_weight / weight)
                    .max(limit - current)
                    .min(0.0)
            };
            set_main(size, row, current + delta);
            consumed += delta;
        }
        if consumed.abs() <= f32::EPSILON {
            break;
        }
        remaining -= consumed;
    }
}
fn justify(justify: RegularJustifyContent, free: f32, count: usize) -> (f32, f32) {
    let free = free.max(0.0);
    match justify {
        RegularJustifyContent::Start => (0.0, 0.0),
        RegularJustifyContent::End => (free, 0.0),
        RegularJustifyContent::Center => (free / 2.0, 0.0),
        RegularJustifyContent::SpaceBetween if count > 1 => (0.0, free / (count - 1) as f32),
        RegularJustifyContent::SpaceAround if count > 0 => {
            (free / count as f32 / 2.0, free / count as f32)
        }
        RegularJustifyContent::SpaceEvenly if count > 0 => {
            (free / (count + 1) as f32, free / (count + 1) as f32)
        }
        _ => (0.0, 0.0),
    }
}
fn main(size: [f32; 2], row: bool) -> f32 {
    if row {
        size[0]
    } else {
        size[1]
    }
}
fn cross(size: [f32; 2], row: bool) -> f32 {
    if row {
        size[1]
    } else {
        size[0]
    }
}
fn set_main(size: &mut [f32; 2], row: bool, value: f32) {
    if row {
        size[0] = value;
    } else {
        size[1] = value;
    }
}
fn transform_and_round(
    origin: [f32; 2],
    rect: LayoutRect,
    transform: [f32; 6],
    scale: f32,
) -> LayoutRect {
    let [a, _, _, d, tx, ty] = transform;
    let x = origin[0] + rect.x;
    let y = origin[1] + rect.y;
    let round = |value: f32| (value * scale).round() / scale;
    let left = round(x * a + tx);
    let top = round(y * d + ty);
    let right = round((x + rect.width) * a + tx);
    let bottom = round((y + rect.height) * d + ty);
    LayoutRect {
        x: left,
        y: top,
        width: (right - left).max(0.0),
        height: (bottom - top).max(0.0),
    }
}
fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn push_f32(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containment::resolve_estimated_style;
    use crate::taffy_tree::{definite, LayoutError, LayoutTree, TaffySize};
    fn input(count: usize) -> RegularLayoutInput {
        RegularLayoutInput {
            origin: [0.5, 1.0],
            container_size: [320.0, 80.0],
            padding: [8.0, 4.0, 8.0, 4.0],
            gap: 3.0,
            axis: RegularAxis::Row,
            justify_content: RegularJustifyContent::Start,
            align_items: RegularAlignItems::Center,
            rounding_scale: 1.0,
            items: (0..count)
                .map(|_| RegularLayoutItem::fixed(24.0, 20.0))
                .collect(),
        }
    }
    #[test]
    fn empty_content_is_an_explicit_fallback() {
        assert_eq!(
            input(0).compute_cpu(),
            Err(RegularLayoutFallback::EmptyContent)
        );
    }
    #[test]
    fn packing_uses_stable_records() {
        let packed = input(2).pack().expect("valid regular input");
        assert_eq!(packed.params.len(), REGULAR_LAYOUT_PARAMS_SIZE);
        assert_eq!(packed.items.len(), 2 * REGULAR_LAYOUT_ITEM_STRIDE);
    }
    #[test]
    fn constraints_gaps_rounding_and_reverse_direction_are_deterministic() {
        let mut layout = input(3);
        layout.axis = RegularAxis::RowReverse;
        layout.items[0].min_size[0] = 30.0;
        layout.items[1].max_size[0] = 20.0;
        layout.items[2].transform = [1.0, 0.0, 0.0, 1.0, 0.25, 0.0];
        let output = layout.compute_cpu().expect("valid regular input");
        assert_eq!(output.len(), 3);
        assert!(output[0].x > output[1].x);
        assert_eq!(output[2].x.fract(), 0.0);
    }
    #[test]
    fn rotated_or_sheared_transforms_fall_back() {
        let mut layout = input(1);
        layout.items[0].transform[1] = 1.0;
        assert_eq!(
            layout.validate(),
            Err(RegularLayoutFallback::UnsupportedTransform)
        );
    }

    #[test]
    fn lowered_line_matches_taffy_for_fixed_children_and_spacing() -> Result<(), LayoutError> {
        let child_style = |width: f32, height: f32| LayoutStyle {
            size: TaffySize {
                width: Dimension::length(width),
                height: Dimension::length(height),
            },
            ..LayoutStyle::default()
        };
        let container = LayoutStyle {
            size: TaffySize {
                width: Dimension::length(320.0),
                height: Dimension::length(80.0),
            },
            padding: crate::taffy_tree::LayoutSides {
                left: LengthPercentage::length(7.0),
                right: LengthPercentage::length(9.0),
                top: LengthPercentage::length(5.0),
                bottom: LengthPercentage::length(3.0),
            },
            border: crate::taffy_tree::LayoutSides {
                left: LengthPercentage::length(1.0),
                right: LengthPercentage::length(2.0),
                top: LengthPercentage::length(1.0),
                bottom: LengthPercentage::length(2.0),
            },
            gap: TaffySize {
                width: LengthPercentage::length(2.5),
                height: LengthPercentage::length(0.0),
            },
            ..LayoutStyle::default()
        };
        let mut children = [
            child_style(24.0, 20.0),
            child_style(30.0, 18.0),
            child_style(16.0, 22.0),
        ];
        children[0].min_size.width = Dimension::length(28.0);
        children[1].max_size.width = Dimension::length(29.0);
        let mut tree = LayoutTree::new();
        let nodes: Vec<_> = children
            .iter()
            .map(|style| tree.request_layout(style.clone(), &[]))
            .collect::<Result<_, _>>()?;
        let root = tree.request_layout(container.clone(), &nodes)?;
        tree.compute_layout(root, definite(320.0, 80.0))?;
        let lowered = lower_styles(&container, &children, tree.layout_of(root)?, 1.0)
            .expect("regular styles");
        let actual = lowered.compute_cpu().expect("regular line");
        let expected: Vec<_> = nodes
            .iter()
            .map(|node| tree.layout_of(*node))
            .collect::<Result<_, _>>()?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn unresolved_or_nested_content_is_never_guessed() {
        let container = LayoutStyle::default();
        let auto_child = LayoutStyle::default();
        assert_eq!(
            lower_styles(
                &container,
                &[auto_child],
                LayoutRect {
                    x: 0.0,
                    y: 0.0,
                    width: 100.0,
                    height: 50.0
                },
                1.0
            ),
            Err(RegularLayoutFallback::UnsupportedContentSize)
        );
        assert_eq!(
            lower_styles_with_nesting(
                &container,
                &[],
                LayoutRect {
                    x: 0.0,
                    y: 0.0,
                    width: 100.0,
                    height: 50.0
                },
                1.0,
                true
            ),
            Err(RegularLayoutFallback::NestedContent)
        );
    }

    #[test]
    fn invalid_estimates_keep_regular_layout_on_the_exact_fallback() {
        let container = LayoutStyle {
            size: TaffySize {
                width: Dimension::length(100.0),
                height: Dimension::length(40.0),
            },
            ..LayoutStyle::default()
        };
        let child = resolve_estimated_style(
            LayoutStyle::default(),
            Some(crate::measure::IntrinsicSize::new(-1.0, 20.0)),
        );
        assert_eq!(
            lower_styles(
                &container,
                &[child],
                LayoutRect {
                    x: 0.0,
                    y: 0.0,
                    width: 100.0,
                    height: 40.0,
                },
                1.0,
            ),
            Err(RegularLayoutFallback::UnsupportedContentSize)
        );
    }

    #[test]
    fn lowered_flex_growth_matches_taffy() -> Result<(), Box<dyn std::error::Error>> {
        let container = LayoutStyle {
            size: TaffySize {
                width: Dimension::length(100.0),
                height: Dimension::length(40.0),
            },
            ..LayoutStyle::default()
        };
        let mut child = LayoutStyle {
            size: TaffySize {
                width: Dimension::length(10.0),
                height: Dimension::length(20.0),
            },
            ..LayoutStyle::default()
        };
        child.flex_grow = 1.0;
        let children = [child.clone(), child];
        let mut tree = LayoutTree::new();
        let nodes: Vec<_> = children
            .iter()
            .map(|style| tree.request_layout(style.clone(), &[]))
            .collect::<Result<_, _>>()?;
        let root = tree.request_layout(container.clone(), &nodes)?;
        tree.compute_layout(root, definite(100.0, 40.0))?;
        let lowered = lower_styles(&container, &children, tree.layout_of(root)?, 1.0)?;
        let actual = lowered.compute_cpu()?;
        let expected: Vec<_> = nodes
            .iter()
            .map(|node| tree.layout_of(*node))
            .collect::<Result<_, _>>()?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
