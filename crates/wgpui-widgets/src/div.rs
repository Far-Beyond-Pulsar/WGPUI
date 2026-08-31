//! `Div` — the element nearly every example in this repository is built out of.
//! See docs/gpu-native-architecture.md §3.4 and §8's Phase 6.6 row.
//!
//! # What Phase 6.6 built here
//!
//! Before this phase `div.rs` and its four submodules were 33 lines in total: a
//! `pub struct Div;` with no fields, no builder, no emission and no layout. Five
//! render pipelines could draw quads, glyph runs, sprites, shadows and
//! underlines byte-exactly, and almost nothing in `2.0` produced any of them,
//! because the element vocabulary that decides *what* to emit did not exist.
//! This file is that vocabulary for the one element that matters most.
//!
//! What is real here: a Tailwind-style builder ([`crate::styled::Styled`]), a
//! resolved paint style that turns into `Quad`/`Shadow` patches
//! ([`interactivity::style::DivStyle`]), children, real Taffy layout through
//! `Description`'s existing `.style()`/`.children()` mechanism, and a
//! `ReconcileKey` split by what a change actually affects
//! ([`diff::DivDiffKey`]).
//!
//! Scroll containers remain outside this interaction seam. `Description::scroll_offset` already carries a
//!   displacement and `.boundary()` already resolves one to a layer transform
//!   (Phase 2), so the *mechanism* exists; what does not is a `ScrollHandle`
//!   deciding what the offset should be, which is `div/scroll_state.rs`.
//!
//! Input state and richer platform interaction remain separate concerns:
//!
//! - `:hover` / `:active` / `:focus`. These need a cascade *above* `DivStyle`
//!   and a hit-test to drive it, and the hit-test needs the input plumbing
//!   §3.4's `div/interactivity/hitbox.rs` is a placeholder for.
//! - Mouse and keyboard event binding (`InteractiveElement`, today's ~456-line
//!   trait block) remains in `div/events.rs`.
//!
//! # Why `describe` consumes `self`
//!
//! A `Description`'s children are `Description`s, and a `Description` is not
//! `Clone` (it owns a `Box<dyn ReconcileKey>` and a `Box<dyn Emit>`). So a `Div`
//! holding built children can only hand them over by move. That matches the
//! legacy shape rather than departing from it: `RenderOnce::render` takes
//! `self` for the same reason, and `AGENTS.md` describes elements as values
//! constructed per frame and turned into a tree once.

//! `Div`, `DivFrameState`/`DivPrepaintState` — the small remainder once
//! `div.rs`'s four seams (event-binding, interactivity, the `Element` impl,
//! scroll/click retained state) move to their own files.
//! See docs/gpu-native-architecture.md §3.4.

pub mod diff;
pub mod events;
pub mod interactivity;
pub mod scroll_state;

use crate::div::diff::DivDiffKey;
use crate::div::interactivity::style::DivStyle;
use crate::styled::Styled;
use wgpui_core::element::Element;
use wgpui_core::patch::emit::{Emission, EmitContext};
use wgpui_core::reconcile::description::{Description, ElementId, TextDecoration, TextOptions};

/// Anything that can become one node of a description tree.
///
/// The 2.0 counterpart of `IntoElement`, reduced to what `wgpui-core` actually
/// consumes. It exists so `div().child(other_div)` reads the way the legacy API
/// does instead of obliging every call site to write `.describe()`; the blanket
/// `Description` impl means an element with its own `describe` still composes
/// without implementing anything.
pub trait IntoDescription {
    /// This value as one description-tree node.
    fn into_description(self) -> Description;
}

impl<T> IntoDescription for T
where
    T: wgpui_core::element::IntoElement,
{
    fn into_description(self) -> Description {
        wgpui_core::element::IntoElement::into_description(self)
    }
}

/// A styled, laid-out box with children — `div()`.
pub struct Div {
    element_id: Option<ElementId>,
    style: DivStyle,
    children: Vec<Description>,
    boundary: bool,
    uncached: bool,
    scroll_offset: [f32; 2],
    estimated_size: Option<[f32; 2]>,
    interaction: events::InteractionState,
    focus_handle: Option<wgpui_core::window::FocusHandle>,
    scroll_handle: Option<scroll_state::ScrollHandle>,
    hover_style: Option<DivStyle>,
    active_style: Option<DivStyle>,
    focus_style: Option<DivStyle>,
    focus_visible_style: Option<DivStyle>,
    group_name: Option<wgpui_text::shaping::SharedString>,
    group_hover_style: Option<(wgpui_text::shaping::SharedString, DivStyle)>,
}

/// A new, unstyled, childless `div`.
///
/// Free function rather than `Div::new`, matching the legacy API exactly: this
/// name appears in every example in the repository and §7 freezes it.
pub fn div() -> Div {
    Div {
        element_id: None,
        style: DivStyle::default(),
        children: Vec::new(),
        boundary: false,
        uncached: false,
        scroll_offset: [0.0, 0.0],
        estimated_size: None,
        interaction: events::InteractionState::new(),
        focus_handle: None,
        scroll_handle: None,
        hover_style: None,
        active_style: None,
        focus_style: None,
        focus_visible_style: None,
        group_name: None,
        group_hover_style: None,
    }
}

impl Default for Div {
    fn default() -> Self {
        div()
    }
}

impl Div {
    pub fn on_click<R: events::IntoEventResult + 'static>(
        mut self,
        handler: impl FnMut(
            &wgpui_core::window::ClickEvent,
            &mut wgpui_core::window::Window,
            &mut wgpui_core::app::App,
        ) -> R
        + 'static,
    ) -> Self {
        self.interaction.on_click(handler);
        self
    }

    pub fn on_mouse_down<R: events::IntoEventResult + 'static>(
        mut self,
        button: wgpui_core::window::MouseButton,
        handler: impl FnMut(
            &wgpui_core::window::InputEvent,
            &mut wgpui_core::window::Window,
            &mut wgpui_core::app::App,
        ) -> R
        + 'static,
    ) -> Self {
        self.interaction.on_mouse_down(button, handler);
        self
    }

    pub fn on_mouse_up<R: events::IntoEventResult + 'static>(
        mut self,
        button: wgpui_core::window::MouseButton,
        handler: impl FnMut(
            &wgpui_core::window::MouseUpEvent,
            &mut wgpui_core::window::Window,
            &mut wgpui_core::app::App,
        ) -> R
            + 'static,
    ) -> Self {
        self.interaction.on_mouse_up(button, handler);
        self
    }

    pub fn on_mouse_move<R: events::IntoEventResult + 'static>(
        mut self,
        handler: impl FnMut(
            &wgpui_core::window::MouseMoveEvent,
            &mut wgpui_core::window::Window,
            &mut wgpui_core::app::App,
        ) -> R
            + 'static,
    ) -> Self {
        self.interaction.on_mouse_move(handler);
        self
    }

    pub fn on_mouse_enter<R: events::IntoEventResult + 'static>(
        mut self,
        handler: impl FnMut(
            &mut wgpui_core::window::Window,
            &mut wgpui_core::app::App,
        ) -> R
            + 'static,
    ) -> Self {
        self.interaction.on_mouse_enter(handler);
        self
    }

    pub fn on_mouse_leave<R: events::IntoEventResult + 'static>(
        mut self,
        handler: impl FnMut(
            &mut wgpui_core::window::Window,
            &mut wgpui_core::app::App,
        ) -> R
            + 'static,
    ) -> Self {
        self.interaction.on_mouse_leave(handler);
        self
    }

    pub fn on_action<A: wgpui_core::action::Action, R: events::IntoEventResult + 'static>(
        mut self,
        handler: impl FnMut(
            &A,
            &mut wgpui_core::window::Window,
            &mut wgpui_core::app::App,
        ) -> R
            + 'static,
    ) -> Self {
        self.interaction.on_action(handler);
        self
    }

    pub fn on_drag<D: 'static, R: 'static>(
        mut self,
        data: D,
        handler: impl FnMut(
            &D,
            [wgpui_core::boundary::Pixels; 2],
            &mut wgpui_core::window::Window,
            &mut wgpui_core::app::App,
        ) -> R
            + 'static,
    ) -> Self {
        self.interaction.on_drag(data, handler);
        self
    }

    pub fn on_drag_hover<D: 'static, R: events::IntoEventResult + 'static>(
        mut self,
        handler: impl FnMut(bool, &mut wgpui_core::window::Window, &mut wgpui_core::app::App) -> R
            + 'static,
    ) -> Self {
        self.interaction.on_drag_hover::<D, R>(handler);
        self
    }

    pub fn on_drop<D: 'static, R: events::IntoEventResult + 'static>(
        mut self,
        handler: impl FnMut(
            &D,
            &mut wgpui_core::window::Window,
            &mut wgpui_core::app::App,
        ) -> R
            + 'static,
    ) -> Self {
        self.interaction.on_drop(handler);
        self
    }

    pub fn on_hover<R: events::IntoEventResult + 'static>(
        mut self,
        handler: impl FnMut(bool, &mut wgpui_core::window::Window, &mut wgpui_core::app::App) -> R
        + 'static,
    ) -> Self {
        self.interaction.on_hover(handler);
        self
    }

    pub fn on_scroll<R: events::IntoEventResult + 'static>(
        mut self,
        handler: impl FnMut(
            &wgpui_core::window::ScrollWheelEvent,
            &mut wgpui_core::window::Window,
            &mut wgpui_core::app::App,
        ) -> R
        + 'static,
    ) -> Self {
        self.interaction.on_scroll(handler);
        self
    }

    pub fn on_key_down<R: events::IntoEventResult + 'static>(
        mut self,
        handler: impl FnMut(
            &wgpui_core::window::KeyDownEvent,
            &mut wgpui_core::window::Window,
            &mut wgpui_core::app::App,
        ) -> R
            + 'static,
    ) -> Self {
        self.interaction.on_key_down(handler);
        self
    }

    pub fn on_key_up<R: events::IntoEventResult + 'static>(
        mut self,
        handler: impl FnMut(
            &wgpui_core::window::KeyUpEvent,
            &mut wgpui_core::window::Window,
            &mut wgpui_core::app::App,
        ) -> R
            + 'static,
    ) -> Self {
        self.interaction.on_key_up(handler);
        self
    }

    pub fn on_text_input<R: events::IntoEventResult + 'static>(
        mut self,
        handler: impl FnMut(
            &wgpui_core::window::TextInputEvent,
            &mut wgpui_core::window::Window,
            &mut wgpui_core::app::App,
        ) -> R
            + 'static,
    ) -> Self {
        self.interaction.on_text_input(handler);
        self
    }

    pub fn on_ime<R: events::IntoEventResult + 'static>(
        mut self,
        handler: impl FnMut(
            &wgpui_core::window::ImeEvent,
            &mut wgpui_core::window::Window,
            &mut wgpui_core::app::App,
        ) -> R
            + 'static,
    ) -> Self {
        self.interaction.on_ime(handler);
        self
    }

    pub fn on_modifiers_changed<R: events::IntoEventResult + 'static>(
        mut self,
        handler: impl FnMut(
            &wgpui_core::window::ModifiersChangedEvent,
            &mut wgpui_core::window::Window,
            &mut wgpui_core::app::App,
        ) -> R
            + 'static,
    ) -> Self {
        self.interaction.on_modifiers_changed(handler);
        self
    }

    pub fn track_focus(mut self, handle: &wgpui_core::window::FocusHandle) -> Self {
        self.focus_handle = Some(*handle);
        self
    }

    /// Resolve a hover style against the element's current resolved style.
    pub fn hover(mut self, apply: impl FnOnce(DivStyle) -> DivStyle) -> Self {
        self.hover_style = Some(apply(self.style.clone()));
        self
    }

    pub fn active(mut self, apply: impl FnOnce(DivStyle) -> DivStyle) -> Self {
        self.active_style = Some(apply(self.style.clone()));
        self
    }

    /// Mark this element as a member of a named hover group.
    pub fn group(mut self, name: impl Into<wgpui_text::shaping::SharedString>) -> Self {
        self.group_name = Some(name.into());
        self
    }

    /// Retain a style to use when a member of `name` is hovered.
    pub fn group_hover(
        mut self,
        name: impl Into<wgpui_text::shaping::SharedString>,
        apply: impl FnOnce(DivStyle) -> DivStyle,
    ) -> Self {
        self.group_hover_style = Some((name.into(), apply(self.style.clone())));
        self
    }

    pub fn focus(mut self, apply: impl FnOnce(DivStyle) -> DivStyle) -> Self {
        self.focus_style = Some(apply(self.style.clone()));
        self
    }

    pub fn focus_visible(mut self, apply: impl FnOnce(DivStyle) -> DivStyle) -> Self {
        self.focus_visible_style = Some(apply(self.style.clone()));
        self
    }

    pub fn is_hovered(&self) -> bool {
        self.interaction.is_hovered()
    }
    pub fn is_active(&self) -> bool {
        self.interaction.is_active()
    }
    pub fn is_focused(&self) -> bool {
        self.interaction.is_focused()
    }

    pub fn handle_input(
        &mut self,
        event: &wgpui_core::window::InputEvent,
        window: &mut wgpui_core::window::Window,
        app: &mut wgpui_core::app::App,
    ) -> wgpui_core::window::EventResult {
        self.interaction.handle_input(event, window, app)
    }

    pub fn update_hover(
        &mut self,
        hovered: bool,
        window: &mut wgpui_core::window::Window,
        app: &mut wgpui_core::app::App,
    ) -> wgpui_core::window::EventResult {
        self.interaction.update_hover(hovered, window, app)
    }

    pub fn track_scroll(mut self, handle: &scroll_state::ScrollHandle) -> Self {
        self.scroll_handle = Some(handle.clone());
        self.boundary = true;
        self
    }

    /// Give this element an explicit identity, so it keeps its instance across a
    /// sibling reorder.
    ///
    /// Optional, and that is the whole point of §4.0: identity is positional by
    /// default (SFD §1.0), so a tree that never calls this still reconciles.
    pub fn id(mut self, element_id: impl Into<ElementId>) -> Self {
        self.element_id = Some(element_id.into());
        self
    }

    /// Append one child.
    pub fn child(mut self, child: impl IntoDescription) -> Self {
        self.children.push(child.into_description());
        self
    }

    /// Append several children.
    pub fn children<C: IntoDescription>(mut self, children: impl IntoIterator<Item = C>) -> Self {
        self.children
            .extend(children.into_iter().map(IntoDescription::into_description));
        self
    }

    /// Make this element a compositing boundary (§4.1).
    pub fn boundary(mut self) -> Self {
        self.boundary = true;
        self
    }

    /// Opt this element and its subtree out of reconciliation (§4.2).
    pub fn uncached(mut self) -> Self {
        self.uncached = true;
        self
    }

    /// Displace this element's children by `offset` without attaching a
    /// scroll handle. Use [`Self::track_scroll`] when input and clamping are
    /// needed as well.
    pub fn scroll_offset(mut self, offset: [f32; 2]) -> Self {
        self.scroll_offset = offset;
        self
    }

    /// Supply a cheap intrinsic estimate for unresolved dimensions. The
    /// estimate is used only for dimensions that remain `auto`; explicit
    /// author sizing always wins. Keeping it on the description makes the
    /// fallback deterministic and avoids invoking a content measurer twice.
    pub fn estimated_size(mut self, size: [f32; 2]) -> Self {
        self.estimated_size = Some([size[0].max(0.0), size[1].max(0.0)]);
        self
    }

    pub fn intrinsic_size(&self) -> Option<[f32; 2]> {
        self.estimated_size
    }

    /// This `div`'s resolved style, for tests and for an inspector.
    pub fn div_style(&self) -> &DivStyle {
        &self.style
    }

    /// How many children this `div` holds.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// This frame's fingerprint.
    pub fn diff_key(&self) -> DivDiffKey {
        DivDiffKey::with_estimate(self.style.clone(), self.children.len(), self.estimated_size)
    }

    /// The per-frame description of this `div` and its subtree.
    pub fn describe(self) -> Description {
        let Div {
            element_id,
            style,
            children,
            mut boundary,
            uncached,
            scroll_offset,
            estimated_size,
            scroll_handle,
            focus_handle,
            hover_style,
            active_style,
            focus_style,
            focus_visible_style,
            group_name: _,
            group_hover_style,
            mut interaction,
            ..
        } = self;

        let style = if interaction.is_focus_visible() {
            focus_visible_style.or(focus_style).unwrap_or(style)
        } else if interaction.is_focused() {
            focus_style.unwrap_or(style)
        } else if interaction.is_active() {
            active_style.unwrap_or(style)
        } else if interaction.is_hovered() {
            hover_style
                .or_else(|| group_hover_style.map(|(_, style)| style))
                .unwrap_or(style)
        } else {
            style
        };
        let key = DivDiffKey::with_estimate(style.clone(), children.len(), estimated_size);
        let mut layout_style = style.layout.clone();
        if let Some([width, height]) = estimated_size {
            if layout_style.size.width == wgpui_layout::taffy_tree::Dimension::auto() {
                layout_style.size.width = wgpui_layout::taffy_tree::Dimension::length(width);
            }
            if layout_style.size.height == wgpui_layout::taffy_tree::Dimension::auto() {
                layout_style.size.height = wgpui_layout::taffy_tree::Dimension::length(height);
            }
        }
        let paint = style;
        let text_options = TextOptions {
            size: paint.text_size,
            line_height: paint.text_line_height,
            color: paint.text_color,
            weight: paint.text_weight.map(|weight| weight.0 as u16),
            italic: paint.text_italic.then_some(true),
            alignment: (paint.text_alignment != 0).then_some(paint.text_alignment),
            nowrap: paint.text_white_space_nowrap.then_some(true),
            ellipsis: paint.text_ellipsis.then_some(true),
            line_clamp: paint.text_line_clamp,
            letter_spacing: paint.text_letter_spacing,
            underline: paint.text_underline.map(|decoration| TextDecoration {
                thickness: decoration.thickness,
                color: decoration.color,
                wavy: decoration.wavy,
            }),
            strikethrough: paint.text_strikethrough.map(|decoration| TextDecoration {
                thickness: decoration.thickness,
                color: decoration.color,
                wavy: false,
            }),
            gradient: paint.text_gradient.clone(),
            gradient_angle: paint.text_gradient_angle,
        };
        let clips_children = matches!(
            layout_style.overflow.x,
            wgpui_layout::taffy_tree::Overflow::Hidden | wgpui_layout::taffy_tree::Overflow::Scroll
        ) || matches!(
            layout_style.overflow.y,
            wgpui_layout::taffy_tree::Overflow::Hidden | wgpui_layout::taffy_tree::Overflow::Scroll
        ) || scroll_handle.is_some();

        if let Some(handle) = scroll_handle.as_ref() {
            boundary = true;
            let handle = handle.clone();
            interaction.on_scroll(move |event, _, _| handle.scroll_wheel(event));
        }

        let scroll_offset = scroll_handle
            .as_ref()
            .map(|handle| [handle.offset().x.value(), handle.offset().y.value()])
            .unwrap_or(scroll_offset);
        let mut description = Description::new::<Div>()
            .diff_key(key)
            .style(layout_style)
            .text_metrics(paint.text_size, paint.text_color)
            .text_options(text_options)
            .scroll_offset(scroll_offset)
            .children(children);

        if clips_children {
            description = description.clip_children();
        }

        if let Some(handle) = scroll_handle.as_ref() {
            let content = estimated_size;
            let handle = handle.clone();
            description =
                description.on_layout_with_content_changed(move |bounds, content_bounds| {
                    let viewport = wgpui_core::geometry::Bounds::new(
                        wgpui_core::geometry::Point::new(
                            wgpui_core::geometry::Pixels(bounds.x),
                            wgpui_core::geometry::Pixels(bounds.y),
                        ),
                        wgpui_core::geometry::Size::pixels(bounds.width, bounds.height),
                    );
                    let measured_content = wgpui_core::geometry::Size::pixels(
                        content_bounds.width.max(viewport.size.width.value()),
                        content_bounds.height.max(viewport.size.height.value()),
                    );
                    let content = content.map_or(measured_content, |size| {
                        wgpui_core::geometry::Size::pixels(
                            measured_content.width.value().max(size[0]),
                            measured_content.height.value().max(size[1]),
                        )
                    });
                    handle.set_viewport(viewport, content)
                });
        }

        if let Some(element_id) = element_id {
            description = description.id(element_id);
        }
        if boundary {
            description = description.boundary();
        }
        if uncached {
            description = description.uncached();
        }
        if let Some(interaction) = interaction.into_description_interaction(focus_handle) {
            description = description.interaction(interaction);
        }

        // An element that paints nothing gets no emitter at all rather than one
        // that writes an empty emission. The distinction is load-bearing:
        // `Emitter::emit` counts an element with an emitter as visited-and-
        // skipped and an element without one as not emitting, and a grouping
        // `div` — most of a real tree — should be the second.
        if paint.primitive_count() == 0 {
            return description;
        }
        description.emit(move |context: &EmitContext, emission: &mut Emission| {
            paint.paint(context.bounds, emission);
        })
    }
}

pub trait StatefulDiv {
    fn on_click<R: events::IntoEventResult + 'static>(
        self,
        handler: impl FnMut(
            &wgpui_core::window::ClickEvent,
            &mut wgpui_core::window::Window,
            &mut wgpui_core::app::App,
        ) -> R
        + 'static,
    ) -> Self;
}

impl StatefulDiv for wgpui_core::element::Stateful<Div> {
    fn on_click<R: events::IntoEventResult + 'static>(
        self,
        handler: impl FnMut(
            &wgpui_core::window::ClickEvent,
            &mut wgpui_core::window::Window,
            &mut wgpui_core::app::App,
        ) -> R
        + 'static,
    ) -> Self {
        self.map_inner(|element| element.on_click(handler))
    }
}

impl Element for Div {
    fn into_description(self) -> Description {
        self.describe()
    }
}

impl Styled for Div {
    fn style(&mut self) -> &mut DivStyle {
        &mut self.style
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::div::interactivity::style::{Corners, Edges};
    use wgpui_core::invalidation::request::FrameSignals;
    use wgpui_core::patch::apply::apply;
    use wgpui_core::patch::emit::{EmitError, Emitter};
    use wgpui_core::reconcile::instance::InstanceKey;
    use wgpui_core::reconcile::plan::NodeOutcome;
    use wgpui_core::reconcile::reconciler::{ReconcileError, Reconciler};
    use wgpui_core::scene::Scene;
    use wgpui_core::scene::layer::{BoundaryId, LayerId, LayerKey};
    use wgpui_layout::taffy_tree::{LayoutTree, definite};

    const VIEWPORT: [f32; 2] = [400.0, 300.0];

    #[derive(Debug)]
    enum FrameError {
        Reconcile(ReconcileError),
        Emit(EmitError),
        Patch(wgpui_core::patch::PatchError),
    }

    impl From<ReconcileError> for FrameError {
        fn from(error: ReconcileError) -> Self {
            FrameError::Reconcile(error)
        }
    }
    impl From<EmitError> for FrameError {
        fn from(error: EmitError) -> Self {
            FrameError::Emit(error)
        }
    }
    impl From<wgpui_core::patch::PatchError> for FrameError {
        fn from(error: wgpui_core::patch::PatchError) -> Self {
            FrameError::Patch(error)
        }
    }

    /// Everything one window holds across frames, so these tests drive real
    /// frames rather than asserting on intermediate values.
    struct Window {
        reconciler: Reconciler,
        layout: LayoutTree,
        emitter: Emitter,
        scene: Scene,
    }

    impl Window {
        fn new() -> Self {
            Self {
                reconciler: Reconciler::new(),
                layout: LayoutTree::new(),
                emitter: Emitter::new(),
                scene: Scene::new(),
            }
        }

        fn draw(&mut self, root: Div) -> Result<Frame, FrameError> {
            let plan = self
                .reconciler
                .reconcile(root.describe(), &mut self.layout)?;
            let node = plan
                .root()
                .map(|node| node.layout_node)
                .ok_or(EmitError::MalformedPlan { index: 0, depth: 0 })?;
            self.layout
                .compute_layout(node, definite(VIEWPORT[0], VIEWPORT[1]))
                .map_err(EmitError::from)?;
            let emission =
                self.emitter
                    .emit(&plan, &self.layout, &FrameSignals::new(), &mut self.scene)?;
            apply(&mut self.scene, &emission.patch)?;
            Ok(Frame {
                outcomes: plan
                    .nodes()
                    .iter()
                    .map(|node| (node.address, node.outcome))
                    .collect(),
                emitted: emission.stats.nodes_emitted,
                updated: emission.stats.records_updated,
                inserted: emission.stats.records_inserted,
                layout_created: self.layout.stats().nodes_created,
                layout_reused: self.layout.stats().nodes_reused,
            })
        }
    }

    struct Frame {
        outcomes: Vec<(InstanceKey, NodeOutcome)>,
        emitted: usize,
        updated: usize,
        inserted: usize,
        layout_created: usize,
        layout_reused: usize,
    }

    impl Frame {
        fn outcome_at(&self, path: &[ElementId]) -> Option<NodeOutcome> {
            let address = InstanceKey::from_path(path);
            self.outcomes
                .iter()
                .find(|(key, _)| *key == address)
                .map(|(_, outcome)| *outcome)
        }
    }

    fn root_layer() -> LayerId {
        LayerId::from_key(LayerKey::untiled(BoundaryId::ROOT))
    }

    /// The root layer's resident quads, in paint order.
    fn quads(window: &Window) -> Vec<wgpui_core::patch::primitive::Quad> {
        window
            .scene
            .quads
            .keys(root_layer())
            .into_iter()
            .filter_map(|key| window.scene.quads.get(root_layer(), key).copied())
            .collect()
    }

    fn card() -> Div {
        div()
            .w(200.0)
            .h(120.0)
            .bg([0.2, 0.3, 0.4, 1.0])
            .border_color([1.0, 1.0, 1.0, 1.0])
            .border_1()
            .rounded_md()
    }

    #[test]
    fn a_styled_childless_div_emits_its_background_and_border() -> Result<(), FrameError> {
        let mut window = Window::new();
        window.draw(card())?;

        assert_eq!(window.scene.quads.len(root_layer()), 2);
        let quads = quads(&window);
        assert_eq!(quads[0].background, [0.2, 0.3, 0.4, 1.0]);
        assert_eq!(quads[0].size, [200.0, 120.0]);
        assert_eq!(quads[0].corner_radii, [6.0; 4]);
        assert_eq!(quads[1].border_widths, [1.0; 4]);
        assert_eq!(quads[1].border_color, [1.0, 1.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn an_unstyled_div_emits_nothing_and_still_lays_out() -> Result<(), FrameError> {
        let mut window = Window::new();
        let frame = window.draw(div().w(100.0).h(100.0).child(card()))?;
        assert_eq!(
            window.scene.quads.len(root_layer()),
            2,
            "the grouping div contributes no primitives of its own"
        );
        assert_eq!(frame.emitted, 1);
        assert_eq!(
            frame.layout_created, 2,
            "it still gets a Taffy node, because its children hang off it"
        );
        Ok(())
    }

    #[test]
    fn an_identical_second_frame_reuses_everything_and_uploads_nothing() -> Result<(), FrameError> {
        let mut window = Window::new();
        window.draw(card())?;
        let settled = window.draw(card())?;
        assert_eq!(
            settled.outcome_at(&[ElementId::Slot(0)]),
            Some(NodeOutcome::Reused)
        );
        assert_eq!(settled.emitted, 0, "a clean, unmoved div must not re-emit");
        assert_eq!(settled.inserted, 0);
        assert_eq!(settled.updated, 0);
        assert_eq!(settled.layout_created, 0);
        assert_eq!(settled.layout_reused, 1);
        Ok(())
    }

    #[test]
    fn a_recolour_updates_the_same_records_in_place() -> Result<(), FrameError> {
        let mut window = Window::new();
        window.draw(card())?;
        let keys = window.scene.quads.keys(root_layer());

        let recoloured = window.draw(card().bg([1.0, 0.0, 0.0, 1.0]))?;
        assert_eq!(recoloured.updated, 2, "two quads, each at its own ordinal");
        assert_eq!(recoloured.inserted, 0);
        assert_eq!(
            window.scene.quads.keys(root_layer()),
            keys,
            "a stable emission order means stable per-primitive addresses (§5.0)"
        );
        assert_eq!(quads(&window)[0].background, [1.0, 0.0, 0.0, 1.0]);
        Ok(())
    }

    #[test]
    fn adding_a_background_inserts_a_record_rather_than_updating_one() -> Result<(), FrameError> {
        let mut window = Window::new();
        let bare = || div().w(200.0).h(120.0).border_color([1.0; 4]).border_1();
        window.draw(bare())?;
        assert_eq!(window.scene.quads.len(root_layer()), 1);

        // The background is emitted *before* the border, so gaining one shifts
        // the border to ordinal 1 — an insert plus an update, not one insert.
        let grown = window.draw(bare().bg([0.0, 1.0, 0.0, 1.0]))?;
        assert_eq!(window.scene.quads.len(root_layer()), 2);
        assert_eq!(grown.inserted, 1);
        assert_eq!(grown.updated, 1);
        let quads = quads(&window);
        assert_eq!(quads[0].background, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(quads[1].border_widths, [1.0; 4]);
        Ok(())
    }

    #[test]
    fn the_builder_reaches_every_field_the_paint_path_reads() {
        let styled = div()
            .bg([0.1, 0.2, 0.3, 0.4])
            .border_color([0.5, 0.6, 0.7, 0.8])
            .border_b(3.0)
            .rounded_t(9.0)
            .shadow_sm();
        let style = styled.div_style();
        assert_eq!(style.background, Some([0.1, 0.2, 0.3, 0.4]));
        assert_eq!(style.border_color, Some([0.5, 0.6, 0.7, 0.8]));
        assert_eq!(
            style.border_widths,
            Edges {
                bottom: 3.0,
                ..Edges::default()
            }
        );
        assert_eq!(
            style.corner_radii,
            Corners {
                top_left: 9.0,
                top_right: 9.0,
                ..Corners::default()
            }
        );
        assert_eq!(style.box_shadow.len(), 2);
        assert_eq!(
            style.primitive_count(),
            4,
            "two shadow layers, one background, one border"
        );
    }

    #[test]
    fn an_explicit_id_survives_a_sibling_reorder_and_a_positional_one_does_not()
    -> Result<(), FrameError> {
        let mut window = Window::new();
        let tree = |swapped: bool| {
            let first = card().id("first");
            let second = card().id("second").bg([1.0, 0.0, 0.0, 1.0]);
            let row = div().w(400.0).h(300.0).flex_row();
            if swapped {
                row.child(second).child(first)
            } else {
                row.child(first).child(second)
            }
        };
        window.draw(tree(false))?;
        let swapped = window.draw(tree(true))?;
        assert_eq!(
            swapped.outcome_at(&[ElementId::Slot(0), ElementId::from("first")]),
            Some(NodeOutcome::Reused),
            "a named child must keep its instance across a reorder"
        );
        assert_eq!(
            swapped.outcome_at(&[ElementId::Slot(0), ElementId::from("second")]),
            Some(NodeOutcome::Reused)
        );
        Ok(())
    }
}
