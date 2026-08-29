//! Style application + `classify_style_change` (§6.2's engine).
//! See docs/gpu-native-architecture.md §3.4, §6.2, and §8's Phase 6.6 row.
//!
//! # What this file is, and what it deliberately is not
//!
//! It is the 2.0 counterpart of the *paint* half of `src/style.rs` — the
//! ~145-line `Style::paint` that turns a resolved style into `paint_shadows` and
//! `paint_quad` calls — plus the style storage those fields live in. It is
//! transcribed from that function rather than reinvented, because §8's Phase 6.6
//! gate is byte-exactness against the legacy renderer and the sequence of quads
//! is as much a part of that as the shader is.
//!
//! It is **not** the legacy `StyleRefinement`. A refinement is a sparse
//! `Option`-per-field overlay whose whole purpose is cascading — a base style
//! refined by `:hover`, then by `:active`, then by a group state. §8's Phase 6.6
//! row scopes interactive states out explicitly, and building a cascade
//! machinery with exactly one layer in it would be inventing the shape of a
//! problem this phase does not have. [`DivStyle`] is therefore a *resolved*
//! style: every field has a value, and the one field that is genuinely optional
//! in the legacy type (`background`, `border_color`) stays optional here because
//! "no background" and "a transparent background" reach different branches of
//! `Style::paint` and must keep doing so.
//!
//! When interactive states land, the cascade goes *above* this type — a
//! refinement resolving down to a `DivStyle` — and nothing here changes.
//!
//! # The two concerns are kept apart, as they are in the legacy code
//!
//! `Style` in the legacy crate carries both layout inputs (flex, size, padding)
//! and paint inputs (background, border, radii, shadow), and `Style::paint`
//! reads only the second set while `Style::to_taffy` reads only the first.
//! [`DivStyle`] keeps that separation visible rather than implied: the layout
//! half is a `LayoutStyle` (which *is* Taffy's own type, §3.2), the paint half
//! is the fields beside it, and [`DivStyle::paint`] cannot see the layout half
//! at all because it takes only the resolved rectangle.

use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::patch::emit::Emission;
use wgpui_core::patch::primitive::{Quad, Shadow};
use wgpui_layout::taffy_tree::{LayoutRect, LayoutStyle};

/// A per-corner value, in the legacy `Corners<T>` field order.
///
/// Named fields rather than a bare `[f32; 4]` because the order is a contract
/// with two shaders (`pick_corner_radius` in both the legacy quad shader and
/// 2.0's), and a transposed pair rounds the wrong corner while producing a
/// picture that looks entirely plausible.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Corners {
    /// Top-left radius, in pixels.
    pub top_left: f32,
    /// Top-right radius, in pixels.
    pub top_right: f32,
    /// Bottom-right radius, in pixels.
    pub bottom_right: f32,
    /// Bottom-left radius, in pixels.
    pub bottom_left: f32,
}

impl Corners {
    /// The same radius at every corner.
    pub const fn all(radius: f32) -> Corners {
        Corners {
            top_left: radius,
            top_right: radius,
            bottom_right: radius,
            bottom_left: radius,
        }
    }

    /// In [`Quad::corner_radii`] order.
    pub const fn to_array(self) -> [f32; 4] {
        [
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left,
        ]
    }

    /// Whether every corner is square.
    pub fn is_zero(self) -> bool {
        self.to_array().iter().all(|radius| *radius == 0.0)
    }

    /// The largest radius.
    pub fn max(self) -> f32 {
        self.to_array().iter().copied().fold(0.0, f32::max)
    }

    /// Clamp every radius to half the shorter side, as
    /// `Corners::clamp_radii_for_quad_size` does (`src/geometry.rs:2396`).
    ///
    /// The legacy `Style::paint` applies this before building either quad, so a
    /// `rounded_full()` box becomes a pill rather than folding its corner arcs
    /// through each other. Neither fragment shader clamps, so doing it here is
    /// not an optimisation — it is where the behaviour lives.
    pub fn clamped_for(self, size: [f32; 2]) -> Corners {
        let max = size[0].min(size[1]) / 2.0;
        Corners {
            top_left: self.top_left.min(max),
            top_right: self.top_right.min(max),
            bottom_right: self.bottom_right.min(max),
            bottom_left: self.bottom_left.min(max),
        }
    }
}

/// A per-side value, in the legacy `Edges<T>` field order.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Edges {
    /// Top width, in pixels.
    pub top: f32,
    /// Right width, in pixels.
    pub right: f32,
    /// Bottom width, in pixels.
    pub bottom: f32,
    /// Left width, in pixels.
    pub left: f32,
}

impl Edges {
    /// The same width on every side.
    pub const fn all(width: f32) -> Edges {
        Edges {
            top: width,
            right: width,
            bottom: width,
            left: width,
        }
    }

    /// In [`Quad::border_widths`] order.
    pub const fn to_array(self) -> [f32; 4] {
        [self.top, self.right, self.bottom, self.left]
    }

    /// Whether every side is zero-width.
    pub fn is_zero(self) -> bool {
        self.to_array().iter().all(|width| *width == 0.0)
    }

    /// The widest side.
    pub fn max(self) -> f32 {
        self.to_array().iter().copied().fold(0.0, f32::max)
    }
}

/// One CSS `box-shadow` layer.
///
/// The legacy `BoxShadow` (`src/style.rs:316`) with its `Hsla` swapped for the
/// straight-alpha RGBA every 2.0 primitive carries, and its `Pixels` newtypes
/// dropped for the same reason `wgpui-core` never grew them.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BoxShadow {
    /// Straight-alpha RGBA.
    pub color: [f32; 4],
    /// Displacement from the element's own rectangle.
    pub offset: [f32; 2],
    /// Gaussian sigma.
    pub blur_radius: f32,
    /// How far the shadow's rectangle grows past the element's on every side
    /// before the blur is applied. Negative values shrink it, which is what
    /// every multi-layer Tailwind shadow uses for its second layer.
    pub spread_radius: f32,
}

/// Everything a `div()` needs to lay itself out and to paint itself.
///
/// See this module's doc for why this is a resolved style rather than a
/// refinement, and why the layout and paint halves sit side by side rather than
/// merged.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DivStyle {
    /// The Taffy style this element's layout node is created with.
    pub layout: LayoutStyle,
    /// Fill colour. `None` and `Some(transparent)` are different: the legacy
    /// `Style::paint` skips the background quad for both, but only `None` means
    /// the author never asked for one, and only the author's answer survives a
    /// future cascade.
    pub background: Option<[f32; 4]>,
    /// Border colour. `None` disables the border quad however wide the edges
    /// are, matching `Style::is_border_visible`.
    pub border_color: Option<[f32; 4]>,
    /// Per-side border widths.
    pub border_widths: Edges,
    /// Per-corner radii, before [`Corners::clamped_for`].
    pub corner_radii: Corners,
    /// `box-shadow` layers, painted in order, all *behind* the element.
    pub box_shadow: Vec<BoxShadow>,
    pub text_color: Option<[f32; 4]>,
    pub text_size: Option<f32>,
    pub text_line_height: Option<f32>,
    pub text_weight: Option<wgpui_text::shaping::FontWeight>,
    pub text_italic: bool,
    pub text_alignment: u8,
    pub text_line_through: bool,
    pub text_gradient: Option<Vec<([f32; 4], f32)>>,
    pub text_gradient_angle: Option<f32>,
}

impl DivStyle {
    /// Whether the border quad is painted at all.
    ///
    /// `Style::is_border_visible` (`src/style.rs:829`), transcribed: a border
    /// needs both a non-transparent colour and at least one non-zero side.
    pub fn is_border_visible(&self) -> bool {
        self.border_color.is_some_and(|color| color[3] > 0.0) && !self.border_widths.is_zero()
    }

    /// Whether the background quad is painted at all.
    ///
    /// `Style::paint`'s `background_color.is_some_and(|color| !color.is_transparent())`.
    pub fn is_background_visible(&self) -> bool {
        self.background.is_some_and(|color| color[3] > 0.0)
    }

    /// Write this style's primitives for an element resolved to `bounds`.
    ///
    /// **A transcription of `Style::paint` (`src/style.rs:683`), in its order**,
    /// which is the order the legacy renderer composites in and therefore the
    /// order byte-exactness depends on:
    ///
    /// 1. every `box-shadow` layer, behind everything;
    /// 2. the background quad, if the background is visible;
    /// 3. *the element's children* — which happens outside this function,
    ///    because in 2.0 a child is a separate element the emit walk visits
    ///    after its parent, not a continuation closure this function calls;
    /// 4. the border quad, if the border is visible.
    ///
    /// # Two deliberate departures, both named rather than hidden
    ///
    /// **The border is one quad, not four.** `Style::paint` paints the border by
    /// drawing the *same* full-bounds quad four times, each clipped by a
    /// `ContentMask` to one edge band. 2.0 has no per-primitive content mask
    /// (§5.2 sends the frame's clip to the occlusion pass instead), so it draws
    /// that quad once, unclipped. The two are equal wherever the four bands do
    /// not overlap and together cover every fragment the quad paints — which
    /// they do for any box wider and taller than twice its widest border, since
    /// the shader returns a zero-alpha colour everywhere inside the inner edge
    /// and a zero-alpha `over` is the identity. That is an *argument*, so
    /// `tests/legacy_div_differential.rs` checks it by rendering the legacy
    /// four-draw sequence and comparing pixels, rather than leaving it asserted.
    ///
    /// **Step 3's ordering is inverted relative to legacy.** The legacy border
    /// is painted *after* the children, so a child overflowing its parent is
    /// drawn under the border. In 2.0 a parent's whole emission is appended
    /// before any child's, so the border lands *under* the children instead.
    /// This is invisible for any child that stays inside its parent's padding —
    /// which is every child in a laid-out flex box that does not overflow — and
    /// it is a real difference for one that does not. It is recorded in
    /// `docs/phase-6.6-results.md` as an open item rather than worked around,
    /// because the fix belongs to §5.1's ordering pass (which is what decides
    /// z-order) and not to this function.
    pub fn paint(&self, bounds: LayoutRect, emission: &mut Emission) {
        let origin = [bounds.x, bounds.y];
        let size = [bounds.width, bounds.height];
        let corner_radii = self.corner_radii.clamped_for(size);

        for shadow in &self.box_shadow {
            emission.shadow(self.shadow_primitive(shadow, origin, size, corner_radii));
        }

        if self.is_background_visible() {
            let background = self.background.unwrap_or([0.0; 4]);
            // The legacy background quad carries a border colour that is its own
            // background with the alpha zeroed. That is not decoration: the
            // fragment shader still evaluates `over(background, border_color)`
            // near a rounded corner, and a zero-alpha `above` makes that `over`
            // return the background unchanged. Copying the rgb as well as the
            // alpha is what `Style::paint` does, so it is what is done here.
            let mut border_color = background;
            border_color[3] = 0.0;
            emission.quad(Quad {
                origin,
                size,
                background,
                border_color,
                corner_radii: corner_radii.to_array(),
                border_widths: [0.0; 4],
            });
        }

        if self.is_border_visible() {
            let border_color = self.border_color.unwrap_or([0.0; 4]);
            let mut background = border_color;
            background[3] = 0.0;
            emission.quad(Quad {
                origin,
                size,
                background,
                border_color,
                corner_radii: corner_radii.to_array(),
                border_widths: self.border_widths.to_array(),
            });
        }
    }

    /// One `box-shadow` layer as a [`Shadow`] primitive.
    ///
    /// `Window::paint_shadows` (`src/window.rs:5679`), transcribed: the shadow's
    /// rectangle is the element's, displaced by the layer's offset and then
    /// dilated by its spread radius. `dilate` moves the origin *in* by the
    /// amount and grows the size by twice it (`src/geometry.rs:1060`), so a
    /// negative spread — which every multi-layer Tailwind shadow uses — shrinks
    /// the rectangle, exactly as CSS specifies.
    fn shadow_primitive(
        &self,
        shadow: &BoxShadow,
        origin: [f32; 2],
        size: [f32; 2],
        corner_radii: Corners,
    ) -> Shadow {
        Shadow {
            origin: [
                origin[0] + shadow.offset[0] - shadow.spread_radius,
                origin[1] + shadow.offset[1] - shadow.spread_radius,
            ],
            size: [
                size[0] + 2.0 * shadow.spread_radius,
                size[1] + 2.0 * shadow.spread_radius,
            ],
            color: shadow.color,
            // The *unspread* box's clamped radii, which is what
            // `Window::paint_shadows` is handed: `Style::paint` computes
            // `corner_radii` once against `bounds.size` and passes the same
            // value to every shadow layer, however far that layer's own spread
            // moved its rectangle. Recomputing the clamp against the spread
            // rectangle here would be more principled and would not match.
            corner_radii: corner_radii.to_array(),
            blur_radius: shadow.blur_radius,
        }
    }

    /// How many primitives [`DivStyle::paint`] will write, without writing them.
    ///
    /// Used by tests that want to assert on the emission shape without a scene,
    /// and by [`crate::div::diff::DivDiffKey`]'s doc to explain why a background
    /// appearing or disappearing is a structural change rather than a value one.
    pub fn primitive_count(&self) -> usize {
        self.box_shadow.len()
            + usize::from(self.is_background_visible())
            + usize::from(self.is_border_visible())
    }
}

/// Which invalidation axes a style change raises.
///
/// **§6.2's engine, and the reason `Div`'s key is split where `StyledText`'s is
/// not.** The legacy `classify_style_change` (`src/elements/div.rs:2299`) makes
/// exactly this distinction and it is right: a `div`'s style has a large
/// layout-affecting half (size, flex, padding, margin, position) and a large
/// paint-affecting half (background, border colour, radii, shadow), and a hover
/// recolour must not re-run Taffy.
///
/// Border *widths* are in both halves, deliberately: they change the painted
/// band and, because Taffy lays out against a border box, they change where
/// children go. The legacy comparison lists `border_widths` under its
/// layout-changed test for the same reason.
pub fn classify_style_change(current: &DivStyle, previous: &DivStyle) -> Invalidation {
    let mut axes = Invalidation::empty();

    if current.layout != previous.layout || current.border_widths != previous.border_widths {
        axes |= Invalidation::LAYOUT;
    }
    if current.background != previous.background
        || current.border_color != previous.border_color
        || current.border_widths != previous.border_widths
        || current.corner_radii != previous.corner_radii
        || current.box_shadow != previous.box_shadow
    {
        axes |= Invalidation::DISPLAY;
    }
    axes
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpui_layout::taffy_tree::{Dimension, LayoutSize};

    fn bounds() -> LayoutRect {
        LayoutRect {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 60.0,
        }
    }

    fn styled() -> DivStyle {
        DivStyle {
            background: Some([0.5, 0.5, 0.5, 1.0]),
            border_color: Some([1.0, 0.0, 0.0, 1.0]),
            border_widths: Edges::all(2.0),
            corner_radii: Corners::all(8.0),
            ..DivStyle::default()
        }
    }

    #[test]
    fn a_background_and_a_border_are_two_quads_in_that_order() {
        let mut emission = Emission::new();
        styled().paint(bounds(), &mut emission);
        assert_eq!(emission.quads().len(), 2);
        assert_eq!(emission.shadows().len(), 0);

        let background = emission.quads()[0];
        assert_eq!(background.background, [0.5, 0.5, 0.5, 1.0]);
        assert_eq!(
            background.border_widths, [0.0; 4],
            "the background quad must carry no border, or the border is drawn twice"
        );
        assert_eq!(
            background.border_color,
            [0.5, 0.5, 0.5, 0.0],
            "`Style::paint` zeroes the alpha of the background's own colour"
        );

        let border = emission.quads()[1];
        assert_eq!(border.border_color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(border.border_widths, [2.0; 4]);
        assert_eq!(border.background, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(border.origin, [10.0, 20.0]);
        assert_eq!(border.size, [100.0, 60.0]);
    }

    #[test]
    fn an_invisible_background_or_border_emits_nothing_for_itself() {
        for style in [
            DivStyle {
                background: None,
                ..DivStyle::default()
            },
            DivStyle {
                background: Some([1.0, 0.0, 0.0, 0.0]),
                ..DivStyle::default()
            },
            // A width with no colour, and a colour with no width: legacy's
            // `is_border_visible` needs both, and either alone must draw
            // nothing rather than a black band or an invisible one.
            DivStyle {
                border_widths: Edges::all(4.0),
                ..DivStyle::default()
            },
            DivStyle {
                border_color: Some([1.0, 0.0, 0.0, 1.0]),
                ..DivStyle::default()
            },
        ] {
            let mut emission = Emission::new();
            style.paint(bounds(), &mut emission);
            assert!(emission.is_empty(), "{style:?} must paint nothing");
            assert_eq!(style.primitive_count(), 0);
        }
    }

    #[test]
    fn radii_are_clamped_to_half_the_shorter_side() {
        let style = DivStyle {
            corner_radii: Corners::all(9999.0),
            ..styled()
        };
        let mut emission = Emission::new();
        style.paint(bounds(), &mut emission);
        assert_eq!(
            emission.quads()[0].corner_radii,
            [30.0; 4],
            "`rounded_full` on a 100x60 box is a 30px pill, not a 9999px fold"
        );
    }

    #[test]
    fn a_shadow_layer_is_offset_then_dilated_by_its_spread() {
        let style = DivStyle {
            box_shadow: vec![BoxShadow {
                color: [0.0, 0.0, 0.0, 0.25],
                offset: [0.0, 4.0],
                blur_radius: 6.0,
                spread_radius: -1.0,
            }],
            ..styled()
        };
        let mut emission = Emission::new();
        style.paint(bounds(), &mut emission);
        assert_eq!(emission.shadows().len(), 1);
        let shadow = emission.shadows()[0];
        assert_eq!(shadow.origin, [11.0, 25.0], "offset by (0,4), shrunk by 1");
        assert_eq!(shadow.size, [98.0, 58.0]);
        assert_eq!(shadow.blur_radius, 6.0);
        assert_eq!(shadow.corner_radii, [8.0; 4]);
        assert_eq!(
            style.primitive_count(),
            3,
            "one shadow, one background, one border"
        );
    }

    #[test]
    fn shadows_come_before_the_quads_and_keep_their_declared_order() {
        let layer = |alpha: f32| BoxShadow {
            color: [0.0, 0.0, 0.0, alpha],
            offset: [0.0, 0.0],
            blur_radius: 4.0,
            spread_radius: 0.0,
        };
        let style = DivStyle {
            box_shadow: vec![layer(0.1), layer(0.2)],
            ..styled()
        };
        let mut emission = Emission::new();
        style.paint(bounds(), &mut emission);
        assert_eq!(
            emission
                .shadows()
                .iter()
                .map(|shadow| shadow.color[3])
                .collect::<Vec<_>>(),
            vec![0.1, 0.2],
            "a record's cross-frame address is its ordinal, so declared order \
             has to be emission order"
        );
    }

    #[test]
    fn a_recolour_is_display_only_and_a_resize_is_layout() {
        let base = styled();

        let recoloured = DivStyle {
            background: Some([1.0, 1.0, 1.0, 1.0]),
            ..base.clone()
        };
        assert_eq!(
            classify_style_change(&recoloured, &base),
            Invalidation::DISPLAY,
            "a hover colour must never re-run Taffy"
        );

        let rerounded = DivStyle {
            corner_radii: Corners::all(2.0),
            ..base.clone()
        };
        assert_eq!(
            classify_style_change(&rerounded, &base),
            Invalidation::DISPLAY
        );

        let resized = DivStyle {
            layout: LayoutStyle {
                size: LayoutSize {
                    width: Dimension::length(200.0),
                    height: Dimension::length(100.0),
                },
                ..LayoutStyle::default()
            },
            ..base.clone()
        };
        assert_eq!(
            classify_style_change(&resized, &base),
            Invalidation::LAYOUT,
            "a size change repaints nothing by itself — the quad's own values \
             are unchanged, only where layout puts them"
        );

        let rebordered = DivStyle {
            border_widths: Edges::all(4.0),
            ..base.clone()
        };
        assert_eq!(
            classify_style_change(&rebordered, &base),
            Invalidation::LAYOUT | Invalidation::DISPLAY,
            "a border width is in both halves: it moves the border box and \
             repaints the band"
        );

        assert_eq!(
            classify_style_change(&base, &base),
            Invalidation::empty(),
            "an unchanged style must report nothing stale"
        );
    }
}
