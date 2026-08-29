//! The Tailwind-style builder DSL — 2.0's counterpart of `src/styled.rs`'s
//! `Styled` trait. See docs/gpu-native-architecture.md §7.
//!
//! Not in §3.4's file map, and a deliberate addition for the same reason
//! `reconcile/description.rs` and `patch/emit.rs` were: §3.4's map describes
//! where *today's `div.rs`* splits, and the `Styled` trait does not live in
//! `div.rs` today — it lives in `src/styled.rs`, one directory up, because it is
//! implemented by every element and not by `Div` alone. This file is that file's
//! position, kept.
//!
//! # What this is, honestly, and what §8's Phase 8 still owes
//!
//! **Before Phase 6.6 there was no Tailwind DSL anywhere in `2.0` — not a
//! partial one, not a stub. `crates/wgpui-widgets/src/styled.rs` did not
//! exist.** §7 lists "the `Styled` DSL" among the surfaces that are
//! "byte-for-byte the same," which is a statement about the eventual alias crate
//! (§3.7, Phase 8) and was not a claim that anything in `2.0` presented it yet.
//! Phase 6.3's report said the same thing from the other side: nothing emitted a
//! `Quad` from a style because no style existed to emit from.
//!
//! What this file is, then, is a **hand-written subset**, and the subset is
//! bounded by something concrete rather than by taste: every method here is one
//! `wgpui_core`'s primitive and layout vocabulary can actually honour today. The
//! legacy trait is far larger and most of it is *macro-generated* —
//! `gpui_macros::style_helpers!()`, `margin_style_methods!()`,
//! `padding_style_methods!()`, `border_style_methods!()`,
//! `box_shadow_style_methods!()` and five more, expanding to several hundred
//! methods across a spacing scale, a colour scale, and every side/corner
//! combination. Reproducing that surface faithfully means porting a proc-macro
//! crate, which is Phase 8's alias-crate work and not this phase's; **doing a
//! hand-written half of it and calling the API "unchanged" would be the kind of
//! claim this project's phase reports exist to prevent.**
//!
//! So: the shapes below are the legacy shapes and the names are the legacy
//! names, the numeric values of the Tailwind scales are transcribed from
//! `gpui-macros`' own expansion rather than guessed, and the surface is smaller.
//! `docs/phase-6.6-results.md` says exactly how much smaller.
//!
//! # Colours are straight-alpha RGBA, not `Hsla`
//!
//! The legacy DSL takes `Hsla` everywhere and its shaders convert in the vertex
//! stage. Every 2.0 primitive carries straight-alpha RGBA as a value and the
//! conversion happens before a colour reaches a slot — which is what lets a
//! recolour be a `DISPLAY` update over unchanged geometry (`styled_text.rs`
//! makes the same point for glyph runs). The alias crate is where an `Hsla`
//! argument becomes an RGBA field; putting the conversion here would put a
//! colour space in `wgpui-widgets` that nothing in `2.0` otherwise names.

use crate::div::interactivity::style::{BoxShadow, Corners, CursorStyle, DivStyle, Edges};
use wgpui_core::boundary::policy::Pixels;
use wgpui_layout::taffy_tree::{
    AlignContent, AlignItems, Dimension, Display, FlexDirection, FlexWrap, LayoutStyle,
    LengthPercentage, LengthPercentageAuto, Overflow, Position,
};
use wgpui_text::shaping::FontWeight;

/// One `rem`, in pixels.
///
/// The legacy default `rem_size` (`Window::rem_size`), which is what every
/// `p_4()`/`gap_2()`-style method on the legacy trait multiplies its scale
/// number by. 2.0 has no `Window` for a caller to have changed it on, so the
/// scale methods below resolve against this constant; a `rem`-aware surface is
/// the alias crate's problem, not this file's.
pub const REM: f32 = 16.0;

#[derive(Clone, Debug, PartialEq)]
pub struct LinearColorStop {
    pub color: [f32; 4],
    pub position: f32,
}

pub trait IntoStylePixels {
    fn into_style_pixels(self) -> f32;
}

impl IntoStylePixels for f32 {
    fn into_style_pixels(self) -> f32 {
        self
    }
}

impl IntoStylePixels for Pixels {
    fn into_style_pixels(self) -> f32 {
        self.value()
    }
}

/// A Tailwind spacing step, in pixels: `n / 4` rem.
pub const fn spacing(step: f32) -> f32 {
    step * REM / 4.0
}

/// Elements that carry a [`DivStyle`] and can be styled through this trait.
///
/// The legacy trait's shape exactly: one required method handing out the style
/// memory, and every builder method written against it, so a new element opts
/// into the whole DSL with a two-line impl.
pub trait Styled: Sized {
    /// This element's style memory.
    fn style(&mut self) -> &mut DivStyle;

    /// The Taffy half of this element's style memory.
    fn layout_style(&mut self) -> &mut LayoutStyle {
        &mut self.style().layout
    }

    // ---- display ----------------------------------------------------------

    /// `display: flex`.
    fn flex(mut self) -> Self {
        self.layout_style().display = Display::Flex;
        self
    }

    /// `display: block`.
    fn block(mut self) -> Self {
        self.layout_style().display = Display::Block;
        self
    }

    /// `display: none`.
    fn hidden(mut self) -> Self {
        self.layout_style().display = Display::None;
        self
    }

    /// `flex-direction: row` (and `display: flex`, as the legacy method does).
    fn flex_row(mut self) -> Self {
        self.layout_style().display = Display::Flex;
        self.layout_style().flex_direction = FlexDirection::Row;
        self
    }

    /// `flex-direction: column`.
    fn flex_col(mut self) -> Self {
        self.layout_style().display = Display::Flex;
        self.layout_style().flex_direction = FlexDirection::Column;
        self
    }

    /// `flex-wrap: wrap`.
    fn flex_wrap(mut self) -> Self {
        self.layout_style().flex_wrap = FlexWrap::Wrap;
        self
    }

    /// `flex-wrap: nowrap`.
    fn flex_nowrap(mut self) -> Self {
        self.layout_style().flex_wrap = FlexWrap::NoWrap;
        self
    }

    /// `flex: 1 1 0%`.
    fn flex_1(mut self) -> Self {
        let style = self.layout_style();
        style.flex_grow = 1.0;
        style.flex_shrink = 1.0;
        style.flex_basis = Dimension::percent(0.0);
        self
    }

    /// `flex: 1 1 auto`.
    fn flex_auto(mut self) -> Self {
        let style = self.layout_style();
        style.flex_grow = 1.0;
        style.flex_shrink = 1.0;
        style.flex_basis = Dimension::auto();
        self
    }

    /// `flex: none`.
    fn flex_none(mut self) -> Self {
        let style = self.layout_style();
        style.flex_grow = 0.0;
        style.flex_shrink = 0.0;
        self
    }

    /// `flex-shrink: 0`.
    fn flex_shrink_0(mut self) -> Self {
        self.layout_style().flex_shrink = 0.0;
        self
    }

    /// `flex-grow: <grow>`.
    fn flex_grow(mut self, grow: f32) -> Self {
        self.layout_style().flex_grow = grow;
        self
    }

    fn grid(mut self) -> Self {
        self.layout_style().display = Display::Grid;
        self
    }

    fn grid_cols(mut self, columns: u16) -> Self {
        use wgpui_layout::taffy_tree::{GridTemplateComponent, TrackSizingFunction};
        use wgpui_layout::taffy_tree::FromFr;
        self.layout_style().display = Display::Grid;
        self.layout_style().grid_template_columns = (0..columns)
            .map(|_| GridTemplateComponent::Single(TrackSizingFunction::from_fr(1.0)))
            .collect();
        self
    }

    fn grid_rows(mut self, rows: u16) -> Self {
        use wgpui_layout::taffy_tree::{GridTemplateComponent, TrackSizingFunction};
        use wgpui_layout::taffy_tree::FromFr;
        self.layout_style().display = Display::Grid;
        self.layout_style().grid_template_rows = (0..rows)
            .map(|_| GridTemplateComponent::Single(TrackSizingFunction::from_fr(1.0)))
            .collect();
        self
    }

    fn col_span(mut self, span: u16) -> Self {
        self.layout_style().grid_column = wgpui_layout::taffy_tree::Line {
            start: wgpui_layout::taffy_tree::GridPlacement::Span(span),
            end: wgpui_layout::taffy_tree::GridPlacement::Span(span),
        };
        self
    }

    fn col_span_full(mut self) -> Self {
        self.layout_style().grid_column = wgpui_layout::taffy_tree::Line {
            start: wgpui_layout::taffy_tree::GridPlacement::Line(1.into()),
            end: wgpui_layout::taffy_tree::GridPlacement::Line((-1).into()),
        };
        self
    }

    fn overflow_hidden(mut self) -> Self {
        self.layout_style().overflow.x = Overflow::Hidden;
        self.layout_style().overflow.y = Overflow::Hidden;
        self
    }

    fn overflow_scroll(mut self) -> Self {
        self.layout_style().overflow.x = Overflow::Scroll;
        self.layout_style().overflow.y = Overflow::Scroll;
        self
    }

    fn overflow_y_scroll(mut self) -> Self {
        self.layout_style().overflow.y = Overflow::Scroll;
        self
    }

    // ---- alignment --------------------------------------------------------

    /// `align-items: center`.
    fn items_center(mut self) -> Self {
        self.layout_style().align_items = Some(AlignItems::Center);
        self
    }

    /// `align-items: flex-start`.
    fn items_start(mut self) -> Self {
        self.layout_style().align_items = Some(AlignItems::FlexStart);
        self
    }

    /// `align-items: flex-end`.
    fn items_end(mut self) -> Self {
        self.layout_style().align_items = Some(AlignItems::FlexEnd);
        self
    }

    /// `justify-content: center`.
    fn justify_center(mut self) -> Self {
        self.layout_style().justify_content = Some(AlignContent::Center);
        self
    }

    /// `justify-content: flex-start`.
    fn justify_start(mut self) -> Self {
        self.layout_style().justify_content = Some(AlignContent::FlexStart);
        self
    }

    /// `justify-content: flex-end`.
    fn justify_end(mut self) -> Self {
        self.layout_style().justify_content = Some(AlignContent::FlexEnd);
        self
    }

    /// `justify-content: space-between`.
    fn justify_between(mut self) -> Self {
        self.layout_style().justify_content = Some(AlignContent::SpaceBetween);
        self
    }

    // ---- size -------------------------------------------------------------

    /// `width: <pixels>px`.
    fn w(mut self, pixels: impl IntoStylePixels) -> Self {
        self.layout_style().size.width = Dimension::length(pixels.into_style_pixels());
        self
    }

    /// `height: <pixels>px`.
    fn h(mut self, pixels: impl IntoStylePixels) -> Self {
        self.layout_style().size.height = Dimension::length(pixels.into_style_pixels());
        self
    }

    /// Both dimensions, in pixels.
    fn size(self, pixels: impl IntoStylePixels) -> Self {
        let pixels = pixels.into_style_pixels();
        self.w(pixels).h(pixels)
    }

    /// `width: 100%`.
    fn w_full(mut self) -> Self {
        self.layout_style().size.width = Dimension::percent(1.0);
        self
    }

    /// `height: 100%`.
    fn h_full(mut self) -> Self {
        self.layout_style().size.height = Dimension::percent(1.0);
        self
    }

    /// Both dimensions at 100%.
    fn size_full(self) -> Self {
        self.w_full().h_full()
    }

    /// `min-width: <pixels>px`.
    fn min_w(mut self, pixels: impl IntoStylePixels) -> Self {
        self.layout_style().min_size.width = Dimension::length(pixels.into_style_pixels());
        self
    }

    /// `min-height: <pixels>px`.
    fn min_h(mut self, pixels: impl IntoStylePixels) -> Self {
        self.layout_style().min_size.height = Dimension::length(pixels.into_style_pixels());
        self
    }

    /// `max-width: <pixels>px`.
    fn max_w(mut self, pixels: impl IntoStylePixels) -> Self {
        self.layout_style().max_size.width = Dimension::length(pixels.into_style_pixels());
        self
    }

    /// `max-height: <pixels>px`.
    fn max_h(mut self, pixels: impl IntoStylePixels) -> Self {
        self.layout_style().max_size.height = Dimension::length(pixels.into_style_pixels());
        self
    }

    // ---- spacing ----------------------------------------------------------

    /// `padding: <pixels>px` on every side.
    fn p(mut self, pixels: impl IntoStylePixels) -> Self {
        self.layout_style().padding = uniform_rect(pixels.into_style_pixels());
        self
    }

    /// Horizontal padding.
    fn px(mut self, pixels: impl IntoStylePixels) -> Self {
        let pixels = pixels.into_style_pixels();
        let padding = &mut self.layout_style().padding;
        padding.left = LengthPercentage::length(pixels);
        padding.right = LengthPercentage::length(pixels);
        self
    }

    /// Vertical padding.
    fn py(mut self, pixels: impl IntoStylePixels) -> Self {
        let pixels = pixels.into_style_pixels();
        let padding = &mut self.layout_style().padding;
        padding.top = LengthPercentage::length(pixels);
        padding.bottom = LengthPercentage::length(pixels);
        self
    }

    /// Top padding.
    fn pt(mut self, pixels: impl IntoStylePixels) -> Self {
        self.layout_style().padding.top = LengthPercentage::length(pixels.into_style_pixels());
        self
    }

    /// Bottom padding.
    fn pb(mut self, pixels: impl IntoStylePixels) -> Self {
        self.layout_style().padding.bottom = LengthPercentage::length(pixels.into_style_pixels());
        self
    }

    /// Left padding.
    fn pl(mut self, pixels: impl IntoStylePixels) -> Self {
        self.layout_style().padding.left = LengthPercentage::length(pixels.into_style_pixels());
        self
    }

    /// Right padding.
    fn pr(mut self, pixels: impl IntoStylePixels) -> Self {
        self.layout_style().padding.right = LengthPercentage::length(pixels.into_style_pixels());
        self
    }

    /// `margin: <pixels>px` on every side.
    fn m(mut self, pixels: impl IntoStylePixels) -> Self {
        let pixels = pixels.into_style_pixels();
        let margin = &mut self.layout_style().margin;
        *margin = wgpui_layout::taffy_tree::LayoutSides {
            top: LengthPercentageAuto::length(pixels),
            right: LengthPercentageAuto::length(pixels),
            bottom: LengthPercentageAuto::length(pixels),
            left: LengthPercentageAuto::length(pixels),
        };
        self
    }

    /// `gap: <pixels>px` on both axes.
    fn gap(mut self, pixels: impl IntoStylePixels) -> Self {
        let pixels = pixels.into_style_pixels();
        let style = self.layout_style();
        style.gap.width = LengthPercentage::length(pixels);
        style.gap.height = LengthPercentage::length(pixels);
        self
    }

    /// Horizontal gap.
    fn gap_x(mut self, pixels: impl IntoStylePixels) -> Self {
        self.layout_style().gap.width = LengthPercentage::length(pixels.into_style_pixels());
        self
    }

    /// Vertical gap.
    fn gap_y(mut self, pixels: impl IntoStylePixels) -> Self {
        self.layout_style().gap.height = LengthPercentage::length(pixels.into_style_pixels());
        self
    }

    // ---- position ---------------------------------------------------------

    /// `position: absolute`.
    fn absolute(mut self) -> Self {
        self.layout_style().position = Position::Absolute;
        self
    }

    /// `position: relative`.
    fn relative(mut self) -> Self {
        self.layout_style().position = Position::Relative;
        self
    }

    /// `top: <pixels>px`.
    fn top(mut self, pixels: impl IntoStylePixels) -> Self {
        self.layout_style().inset.top = LengthPercentageAuto::length(pixels.into_style_pixels());
        self
    }

    /// `left: <pixels>px`.
    fn left(mut self, pixels: impl IntoStylePixels) -> Self {
        self.layout_style().inset.left = LengthPercentageAuto::length(pixels.into_style_pixels());
        self
    }

    /// `right: <pixels>px`.
    fn right(mut self, pixels: impl IntoStylePixels) -> Self {
        self.layout_style().inset.right = LengthPercentageAuto::length(pixels.into_style_pixels());
        self
    }

    /// `bottom: <pixels>px`.
    fn bottom(mut self, pixels: impl IntoStylePixels) -> Self {
        self.layout_style().inset.bottom = LengthPercentageAuto::length(pixels.into_style_pixels());
        self
    }

    // ---- paint ------------------------------------------------------------

    /// Fill colour, as straight-alpha RGBA. See this module's doc for why not
    /// `Hsla`.
    fn bg(mut self, color: impl Into<[f32; 4]>) -> Self {
        self.style().background = Some(color.into());
        self
    }

    /// Set the element background alpha without changing its layout.
    fn opacity(mut self, opacity: f32) -> Self {
        let color = self.style().background.unwrap_or([1.0, 1.0, 1.0, 1.0]);
        self.style().background = Some([color[0], color[1], color[2], color[3] * opacity]);
        self
    }

    /// Border colour, as straight-alpha RGBA. A border needs both a colour and
    /// a width to be painted at all.
    fn border_color(mut self, color: impl Into<[f32; 4]>) -> Self {
        self.style().border_color = Some(color.into());
        self
    }

    /// A uniform border width, in pixels.
    fn border(mut self, pixels: impl IntoStylePixels) -> Self {
        let pixels = pixels.into_style_pixels();
        self.style().border_widths = Edges::all(pixels);
        // Taffy lays out against the border box, so a border that is not also
        // declared to the layout style would leave children overlapping it.
        // The legacy `Style::to_taffy` reads `border_widths` for exactly this,
        // which is also why `classify_style_change` puts border width in the
        // layout half.
        self.layout_style().border = uniform_rect(pixels);
        self
    }

    /// A 1px border on every side.
    fn border_1(self) -> Self {
        self.border(1.0)
    }

    /// Render the border as repeating dash-gap segments.
    fn border_dashed(mut self) -> Self {
        self.style().border_dashed = true;
        self
    }

    /// A 2px border on every side.
    fn border_2(self) -> Self {
        self.border(2.0)
    }

    fn border_3(self) -> Self {
        self.border(3.0)
    }

    /// A 4px border on every side.
    fn border_4(self) -> Self {
        self.border(4.0)
    }

    /// Border width on the top side only.
    fn border_t(mut self, pixels: impl IntoStylePixels) -> Self {
        let pixels = pixels.into_style_pixels();
        self.style().border_widths.top = pixels;
        self.layout_style().border.top = LengthPercentage::length(pixels);
        self
    }

    /// Border width on the right side only.
    fn border_r(mut self, pixels: impl IntoStylePixels) -> Self {
        let pixels = pixels.into_style_pixels();
        self.style().border_widths.right = pixels;
        self.layout_style().border.right = LengthPercentage::length(pixels);
        self
    }

    /// Border width on the bottom side only.
    fn border_b(mut self, pixels: impl IntoStylePixels) -> Self {
        let pixels = pixels.into_style_pixels();
        self.style().border_widths.bottom = pixels;
        self.layout_style().border.bottom = LengthPercentage::length(pixels);
        self
    }

    /// Border width on the left side only.
    fn border_l(mut self, pixels: impl IntoStylePixels) -> Self {
        let pixels = pixels.into_style_pixels();
        self.style().border_widths.left = pixels;
        self.layout_style().border.left = LengthPercentage::length(pixels);
        self
    }

    fn border_t_1(self) -> Self { self.border_t(1.0) }
    fn border_r_1(self) -> Self { self.border_r(1.0) }
    fn border_b_1(self) -> Self { self.border_b(1.0) }
    fn border_l_1(self) -> Self { self.border_l(1.0) }
    fn border_y_1(self) -> Self { self.border_t_1().border_b_1() }
    fn border_l_3(self) -> Self { self.border_l(3.0) }

    /// A uniform corner radius, in pixels.
    fn rounded(mut self, pixels: impl IntoStylePixels) -> Self {
        self.style().corner_radii = Corners::all(pixels.into_style_pixels());
        self
    }

    /// Tailwind `rounded-sm`: 0.125rem.
    fn rounded_sm(self) -> Self {
        self.rounded(REM * 0.125)
    }

    /// Tailwind `rounded-md`: 0.375rem.
    fn rounded_md(self) -> Self {
        self.rounded(REM * 0.375)
    }

    /// Tailwind `rounded-lg`: 0.5rem.
    fn rounded_lg(self) -> Self {
        self.rounded(REM * 0.5)
    }

    /// Tailwind `rounded-xl`: 0.75rem.
    fn rounded_xl(self) -> Self {
        self.rounded(REM * 0.75)
    }

    /// Tailwind `rounded-full`.
    ///
    /// The legacy value is a literal `px(9999.)`, which
    /// [`Corners::clamped_for`] then reduces to half the shorter side at paint
    /// time. Kept as the same absurd number rather than resolved here, because
    /// the clamp is what makes it a pill and the clamp needs the resolved size.
    fn rounded_full(self) -> Self {
        self.rounded(9999.0)
    }

    /// Radius on both top corners.
    fn rounded_t(mut self, pixels: f32) -> Self {
        let radii = &mut self.style().corner_radii;
        radii.top_left = pixels;
        radii.top_right = pixels;
        self
    }

    /// Radius on both bottom corners.
    fn rounded_b(mut self, pixels: f32) -> Self {
        let radii = &mut self.style().corner_radii;
        radii.bottom_left = pixels;
        radii.bottom_right = pixels;
        self
    }

    /// Radius on both left corners.
    fn rounded_l(mut self, pixels: f32) -> Self {
        let radii = &mut self.style().corner_radii;
        radii.top_left = pixels;
        radii.bottom_left = pixels;
        self
    }

    /// Radius on both right corners.
    fn rounded_r(mut self, pixels: f32) -> Self {
        let radii = &mut self.style().corner_radii;
        radii.top_right = pixels;
        radii.bottom_right = pixels;
        self
    }

    /// Radius on the top-left corner only.
    fn rounded_tl(mut self, pixels: f32) -> Self {
        self.style().corner_radii.top_left = pixels;
        self
    }

    /// Radius on the top-right corner only.
    fn rounded_tr(mut self, pixels: f32) -> Self {
        self.style().corner_radii.top_right = pixels;
        self
    }

    /// Radius on the bottom-right corner only.
    fn rounded_br(mut self, pixels: f32) -> Self {
        self.style().corner_radii.bottom_right = pixels;
        self
    }

    /// Radius on the bottom-left corner only.
    fn rounded_bl(mut self, pixels: f32) -> Self {
        self.style().corner_radii.bottom_left = pixels;
        self
    }

    /// Set the `box-shadow` layers outright.
    fn shadow<S: Into<BoxShadow>>(mut self, shadows: Vec<S>) -> Self {
        self.style().box_shadow = shadows.into_iter().map(Into::into).collect();
        self
    }

    /// Clear every `box-shadow` layer.
    fn shadow_none(mut self) -> Self {
        self.style().box_shadow.clear();
        self
    }

    /// Tailwind `shadow-2xs`.
    fn shadow_2xs(self) -> Self {
        self.shadow(vec![black(0.05, [0.0, 1.0], 0.0, 0.0)])
    }

    /// Tailwind `shadow-xs`.
    fn shadow_xs(self) -> Self {
        self.shadow(vec![black(0.05, [0.0, 1.0], 2.0, 0.0)])
    }

    /// Tailwind `shadow-sm`.
    fn shadow_sm(self) -> Self {
        self.shadow(vec![
            black(0.1, [0.0, 1.0], 3.0, 0.0),
            black(0.1, [0.0, 1.0], 2.0, -1.0),
        ])
    }

    /// Tailwind `shadow-md`.
    fn shadow_md(self) -> Self {
        self.shadow(vec![
            black(0.1, [0.0, 4.0], 6.0, -1.0),
            black(0.1, [0.0, 2.0], 4.0, -2.0),
        ])
    }

    /// Tailwind `shadow-lg`.
    fn shadow_lg(self) -> Self {
        self.shadow(vec![
            black(0.1, [0.0, 10.0], 15.0, -3.0),
            black(0.1, [0.0, 4.0], 6.0, -4.0),
        ])
    }

    /// Tailwind `shadow-xl`.
    fn shadow_xl(self) -> Self {
        self.shadow(vec![
            black(0.1, [0.0, 20.0], 25.0, -5.0),
            black(0.1, [0.0, 8.0], 10.0, -6.0),
        ])
    }

    /// Tailwind `shadow-2xl`.
    fn shadow_2xl(self) -> Self {
        self.shadow(vec![black(0.25, [0.0, 25.0], 50.0, -12.0)])
    }

    fn text_color(mut self, color: impl Into<[f32; 4]>) -> Self {
        self.style().text_color = Some(color.into());
        self
    }

    fn cursor_pointer(mut self) -> Self {
        self.style().cursor = CursorStyle::Pointer;
        self
    }
    fn cursor_grab(mut self) -> Self {
        self.style().cursor = CursorStyle::Grab;
        self
    }
    fn cursor_crosshair(mut self) -> Self {
        self.style().cursor = CursorStyle::Crosshair;
        self
    }
    fn cursor_not_allowed(mut self) -> Self {
        self.style().cursor = CursorStyle::NotAllowed;
        self
    }

    fn text_gradient_horizontal(mut self, from: LinearColorStop, to: LinearColorStop) -> Self {
        self.style().text_gradient = Some(vec![(from.color, from.position), (to.color, to.position)]);
        self.style().text_gradient_angle = Some(90.0);
        self
    }

    fn text_gradient_vertical(mut self, from: LinearColorStop, to: LinearColorStop) -> Self {
        self.style().text_gradient = Some(vec![(from.color, from.position), (to.color, to.position)]);
        self.style().text_gradient_angle = Some(180.0);
        self
    }

    fn text_size(mut self, size: impl IntoStylePixels) -> Self {
        self.style().text_size = Some(size.into_style_pixels());
        self
    }

    fn text_xs(self) -> Self { self.text_size(12.0) }
    fn text_sm(self) -> Self { self.text_size(14.0) }
    fn text_base(self) -> Self { self.text_size(16.0) }
    fn text_lg(self) -> Self { self.text_size(18.0) }
    fn text_xl(self) -> Self { self.text_size(20.0) }
    fn text_2xl(self) -> Self { self.text_size(24.0) }
    fn text_center(mut self) -> Self { self.style().text_alignment = 1; self }
    fn text_right(mut self) -> Self { self.style().text_alignment = 2; self }
    fn font_weight(mut self, weight: FontWeight) -> Self { self.style().text_weight = Some(weight); self }
    fn italic(mut self) -> Self { self.style().text_italic = true; self }
    fn line_height(mut self, height: impl IntoStylePixels) -> Self {
        self.style().text_line_height = Some(height.into_style_pixels());
        self
    }
    fn line_through(mut self) -> Self { self.style().text_line_through = true; self }

    fn p_0p5(self) -> Self { self.p(2.0) }
    fn p_1(self) -> Self { self.p(4.0) }
    fn p_2(self) -> Self { self.p(8.0) }
    fn p_3(self) -> Self { self.p(12.0) }
    fn p_4(self) -> Self { self.p(16.0) }
    fn p_6(self) -> Self { self.p(24.0) }
    fn px_2(self) -> Self { self.px(8.0) }
    fn px_3(self) -> Self { self.px(12.0) }
    fn px_4(self) -> Self { self.px(16.0) }
    fn py_1(self) -> Self { self.py(4.0) }
    fn py_2(self) -> Self { self.py(8.0) }
    fn gap_1(self) -> Self { self.gap(4.0) }
    fn gap_2(self) -> Self { self.gap(8.0) }
    fn gap_3(self) -> Self { self.gap(12.0) }
    fn gap_4(self) -> Self { self.gap(16.0) }
    fn gap_6(self) -> Self { self.gap(24.0) }
    fn w_16(self) -> Self { self.w(64.0) }
    fn size_16(self) -> Self { self.size(64.0) }
    fn size_8(self) -> Self { self.size(32.0) }
    fn size_10(self) -> Self { self.size(40.0) }
    fn h_6(self) -> Self { self.h(24.0) }
    fn h_8(self) -> Self { self.h(32.0) }
    fn h_20(self) -> Self { self.h(80.0) }
    fn h_24(self) -> Self { self.h(96.0) }
    fn h_32(self) -> Self { self.h(128.0) }
    fn top_0(self) -> Self { self.top(0.0) }
    fn top_2(self) -> Self { self.top(8.0) }
    fn top_4(self) -> Self { self.top(16.0) }
    fn top_6(self) -> Self { self.top(24.0) }
    fn left_2(self) -> Self { self.left(8.0) }
    fn left_4(self) -> Self { self.left(16.0) }
    fn left_6(self) -> Self { self.left(24.0) }
    fn right_0(self) -> Self { self.right(0.0) }
    fn bottom_0(self) -> Self { self.bottom(0.0) }
    fn mt_2(mut self) -> Self {
        self.layout_style().margin.top = LengthPercentageAuto::length(8.0);
        self
    }
    fn mb_5(mut self) -> Self {
        self.layout_style().margin.bottom = LengthPercentageAuto::length(20.0);
        self
    }

    fn p_5(self) -> Self { self.p(20.0) }
    fn p_8(self) -> Self { self.p(32.0) }
    fn p_12(self) -> Self { self.p(48.0) }
    fn px_1(self) -> Self { self.px(4.0) }
    fn py_0p5(self) -> Self { self.py(2.0) }
    fn py_1p5(self) -> Self { self.py(6.0) }
    fn py_3(self) -> Self { self.py(12.0) }
    fn py_12(self) -> Self { self.py(48.0) }
    fn gap_0p5(self) -> Self { self.gap(2.0) }
    fn gap_5(self) -> Self { self.gap(20.0) }
    fn gap_8(self) -> Self { self.gap(32.0) }
    fn mt_1(mut self) -> Self {
        self.layout_style().margin.top = LengthPercentageAuto::length(4.0);
        self
    }
    fn h_2(self) -> Self { self.h(8.0) }
    fn h_10(self) -> Self { self.h(40.0) }
    fn h_12(self) -> Self { self.h(48.0) }
    fn h_16(self) -> Self { self.h(64.0) }
    fn h_48(self) -> Self { self.h(192.0) }
    fn bottom_4(self) -> Self { self.bottom(16.0) }
    fn bottom_8(self) -> Self { self.bottom(32.0) }
    fn left_0(self) -> Self { self.left(0.0) }
    fn left_8(self) -> Self { self.left(32.0) }
    fn inset_0(self) -> Self { self.top(0.0).right(0.0).bottom(0.0).left(0.0) }

    // ---- conditionals -----------------------------------------------------

    /// Apply `then` only when `condition` holds.
    ///
    /// `AGENTS.md` documents this as part of the element vocabulary, and every
    /// real tree uses it, so it is here rather than in `Div` alone.
    fn when(self, condition: bool, then: impl FnOnce(Self) -> Self) -> Self {
        if condition { then(self) } else { self }
    }

    /// Apply `then` only when `value` is present.
    fn when_some<T>(self, value: Option<T>, then: impl FnOnce(Self, T) -> Self) -> Self {
        match value {
            Some(value) => then(self, value),
            None => self,
        }
    }
}

/// The shadow layers every Tailwind preset is built from: pure black at some
/// alpha, offset downwards, blurred, spread inwards.
///
/// Transcribed from `gpui-macros`' `box_shadow_style_methods` expansion, where
/// every preset is `hsla(0., 0., 0., a)` — which converts to RGBA `(0, 0, 0, a)`
/// exactly, so the two backends agree on the byte without a colour-space step.
const fn black(alpha: f32, offset: [f32; 2], blur_radius: f32, spread_radius: f32) -> BoxShadow {
    BoxShadow {
        color: [0.0, 0.0, 0.0, alpha],
        offset,
        blur_radius,
        spread_radius,
    }
}

fn uniform_rect(pixels: f32) -> wgpui_layout::taffy_tree::LayoutSides<LengthPercentage> {
    wgpui_layout::taffy_tree::LayoutSides {
        top: LengthPercentage::length(pixels),
        right: LengthPercentage::length(pixels),
        bottom: LengthPercentage::length(pixels),
        left: LengthPercentage::length(pixels),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The smallest possible implementor, so these tests exercise the trait
    /// rather than `Div`.
    #[derive(Default)]
    struct Styleable(DivStyle);

    impl Styled for Styleable {
        fn style(&mut self) -> &mut DivStyle {
            &mut self.0
        }
    }

    #[test]
    fn the_paint_methods_reach_the_paint_half_and_nothing_else() {
        let styled = Styleable::default()
            .bg([1.0, 0.0, 0.0, 1.0])
            .border_color([0.0, 0.0, 1.0, 1.0])
            .rounded_md();
        assert_eq!(styled.0.background, Some([1.0, 0.0, 0.0, 1.0]));
        assert_eq!(styled.0.corner_radii, Corners::all(6.0));
        assert_eq!(
            styled.0.layout,
            LayoutStyle::default(),
            "a colour or a radius must not touch the layout style, or every \
             recolour re-runs Taffy"
        );
    }

    #[test]
    fn a_border_reaches_both_halves() {
        let styled = Styleable::default().border_2();
        assert_eq!(styled.0.border_widths, Edges::all(2.0));
        assert_eq!(
            styled.0.layout.border,
            uniform_rect(2.0),
            "Taffy lays out against the border box, so a border it was not told \
             about would let children overlap it"
        );
    }

    #[test]
    fn per_side_and_per_corner_methods_touch_only_their_own_side() {
        let styled = Styleable::default().border_b(3.0).rounded_t(8.0);
        assert_eq!(
            styled.0.border_widths,
            Edges {
                bottom: 3.0,
                ..Edges::default()
            }
        );
        assert_eq!(
            styled.0.corner_radii,
            Corners {
                top_left: 8.0,
                top_right: 8.0,
                ..Corners::default()
            }
        );
        assert_eq!(styled.0.layout.border.bottom, LengthPercentage::length(3.0));
        assert_eq!(styled.0.layout.border.top, LengthPercentage::length(0.0));
    }

    #[test]
    fn the_tailwind_shadow_presets_are_the_macros_own_numbers() {
        let styled = Styleable::default().shadow_md();
        assert_eq!(
            styled.0.box_shadow,
            vec![
                BoxShadow {
                    color: [0.0, 0.0, 0.0, 0.1],
                    offset: [0.0, 4.0],
                    blur_radius: 6.0,
                    spread_radius: -1.0,
                },
                BoxShadow {
                    color: [0.0, 0.0, 0.0, 0.1],
                    offset: [0.0, 2.0],
                    blur_radius: 4.0,
                    spread_radius: -2.0,
                },
            ]
        );
        assert!(
            Styleable::default()
                .shadow_md()
                .shadow_none()
                .0
                .box_shadow
                .is_empty()
        );
    }

    #[test]
    fn the_spacing_scale_resolves_against_one_rem() {
        assert_eq!(spacing(1.0), 4.0);
        assert_eq!(spacing(4.0), 16.0);
        let styled = Styleable::default().p(spacing(2.0));
        assert_eq!(styled.0.layout.padding, uniform_rect(8.0));
    }

    #[test]
    fn when_and_when_some_apply_only_on_the_taken_branch() {
        let applied = Styleable::default().when(true, |this| this.bg([1.0; 4]));
        let skipped = Styleable::default().when(false, |this| this.bg([1.0; 4]));
        assert_eq!(applied.0.background, Some([1.0; 4]));
        assert_eq!(skipped.0.background, None);

        let some = Styleable::default().when_some(Some(4.0), Styleable::rounded);
        assert_eq!(some.0.corner_radii, Corners::all(4.0));
        let none = Styleable::default().when_some(None::<f32>, Styleable::rounded);
        assert_eq!(none.0.corner_radii, Corners::default());
    }

    #[test]
    fn legacy_layout_aliases_reach_taffy_and_pixel_arithmetic_is_preserved() {
        let styled = Styleable::default()
            .grid_cols(3)
            .overflow_y_scroll()
            .border_l_3()
            .text_xs();
        assert_eq!(styled.0.layout.display, Display::Grid);
        assert_eq!(styled.0.layout.grid_template_columns.len(), 3);
        assert_eq!(styled.0.layout.overflow.y, Overflow::Scroll);
        assert_eq!(styled.0.border_widths.left, 3.0);
        assert_eq!(styled.0.text_size, Some(12.0));
    }
}
