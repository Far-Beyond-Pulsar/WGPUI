use std::cell::RefCell;
use std::rc::Rc;

use super::anchored::{Anchor, AnchorHandle, AnchoredPosition};
use wgpui_core::element::{Element, IntoElement};
use wgpui_core::geometry::{Bounds, Pixels, Point, Rect, Size};
use wgpui_core::invalidation::axes::Invalidation;
use wgpui_core::reconcile::description::{Description, DescriptionInteraction, ElementId};
use wgpui_core::reconcile::diff_key::ReconcileKey;
use wgpui_core::window::{EventResult, FocusHandle, InputEvent};
use wgpui_layout::taffy_tree::{
    Dimension, LayoutSize, LayoutStyle, LengthPercentageAuto, Position,
};

struct OverlayHandleState {
    open: bool,
    revision: u64,
    invalidator: Option<Box<dyn Fn()>>,
    dismissed: Option<Box<dyn FnMut()>>,
}

struct OverlayDismissLayer;

/// Controls one overlay's lifetime without retaining its per-frame content.
#[derive(Clone)]
pub struct OverlayHandle {
    state: Rc<RefCell<OverlayHandleState>>,
}

impl OverlayHandle {
    pub fn new() -> Self {
        Self {
            state: Rc::new(RefCell::new(OverlayHandleState {
                open: true,
                revision: 0,
                invalidator: None,
                dismissed: None,
            })),
        }
    }

    pub fn is_open(&self) -> bool {
        self.state.borrow().open
    }

    pub fn revision(&self) -> u64 {
        self.state.borrow().revision
    }

    pub fn open(&self) {
        self.set_open(true);
    }

    pub fn close(&self) {
        self.set_open(false);
    }

    pub fn toggle(&self) {
        self.set_open(!self.is_open());
    }

    /// Connect state changes to the public native boundary. A WGPU caller can
    /// pass a closure that invokes its public window handle's redraw request.
    pub fn invalidate_with(&self, invalidator: impl Fn() + 'static) {
        self.state.borrow_mut().invalidator = Some(Box::new(invalidator));
    }

    pub fn invalidate(&self) {
        let mut state = self.state.borrow_mut();
        state.revision = state.revision.wrapping_add(1);
        drop(state);
        self.notify_invalidator();
    }

    pub fn on_dismiss(&self, callback: impl FnMut() + 'static) {
        self.state.borrow_mut().dismissed = Some(Box::new(callback));
    }

    /// Close the overlay because an outside click or Escape requested it.
    pub fn dismiss(&self) -> bool {
        if !self.is_open() {
            return false;
        }
        self.set_open(false);
        let mut callback = self.state.borrow_mut().dismissed.take();
        if let Some(callback) = callback.as_deref_mut() {
            callback();
        }
        self.state.borrow_mut().dismissed = callback;
        true
    }

    fn set_open(&self, open: bool) {
        let changed = {
            let mut state = self.state.borrow_mut();
            if state.open == open {
                false
            } else {
                state.open = open;
                state.revision = state.revision.wrapping_add(1);
                true
            }
        };
        if changed {
            self.notify_invalidator();
        }
    }

    fn notify_invalidator(&self) {
        let callback = self.state.borrow_mut().invalidator.take();
        if let Some(callback) = callback {
            callback();
            self.state.borrow_mut().invalidator = Some(callback);
        }
    }
}

impl Default for OverlayHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// One retained overlay entry. Its content is consumed into the ordinary
/// description tree and is reconciled and patched like any other subtree.
pub struct DeferredOverlay {
    id: ElementId,
    content: Option<Description>,
    position: AnchoredPosition,
    anchor_handle: Option<AnchorHandle>,
    viewport: Rect,
    size: Size<Pixels>,
    margin: Pixels,
    z_index: i32,
    handle: OverlayHandle,
    focus_handle: Option<FocusHandle>,
    dismiss_on_escape: bool,
    dismiss_on_click_outside: bool,
}

impl DeferredOverlay {
    pub fn new(id: impl Into<ElementId>, content: impl IntoElement) -> Self {
        Self::from_description(id, content.into_description())
    }

    pub fn from_description(id: impl Into<ElementId>, content: Description) -> Self {
        Self {
            id: id.into(),
            content: Some(content),
            position: AnchoredPosition::new(Bounds::default(), Anchor::TopLeft),
            anchor_handle: None,
            viewport: Rect::EMPTY,
            size: Size::default(),
            margin: Pixels::ZERO,
            z_index: 0,
            handle: OverlayHandle::new(),
            focus_handle: None,
            dismiss_on_escape: true,
            dismiss_on_click_outside: false,
        }
    }

    pub fn empty(id: impl Into<ElementId>) -> Self {
        let mut overlay = Self::from_description(id, Description::new::<()>());
        overlay.content = None;
        overlay
    }

    pub fn id(&self) -> &ElementId {
        &self.id
    }

    pub fn handle(&self) -> OverlayHandle {
        self.handle.clone()
    }

    pub fn anchor(mut self, bounds: Bounds<Pixels>, anchor: Anchor) -> Self {
        self.position = AnchoredPosition::new(bounds, anchor);
        self.anchor_handle = None;
        self
    }

    pub fn track_anchor(mut self, handle: AnchorHandle, anchor: Anchor) -> Self {
        self.position = handle.position(anchor);
        self.anchor_handle = Some(handle);
        self
    }

    pub fn offset(mut self, offset: Point<Pixels>) -> Self {
        self.position.offset = offset;
        self
    }

    pub fn viewport(mut self, viewport: Rect) -> Self {
        self.viewport = viewport;
        self
    }

    pub fn size(mut self, size: Size<Pixels>) -> Self {
        self.size = size;
        self
    }

    pub fn margin(mut self, margin: Pixels) -> Self {
        self.margin = margin;
        self
    }

    pub fn z_index(mut self, z_index: i32) -> Self {
        self.z_index = z_index;
        self
    }

    pub fn focus_handle(mut self, focus_handle: FocusHandle) -> Self {
        self.focus_handle = Some(focus_handle);
        self
    }

    pub fn dismiss_on_escape(mut self, enabled: bool) -> Self {
        self.dismiss_on_escape = enabled;
        self
    }

    pub fn dismiss_on_click_outside(mut self, enabled: bool) -> Self {
        self.dismiss_on_click_outside = enabled;
        self
    }

    pub fn controller(mut self, handle: OverlayHandle) -> Self {
        self.handle = handle;
        self
    }

    pub fn is_renderable(&self) -> bool {
        self.handle.is_open() && self.content.is_some()
    }

    pub fn resolved_position(&self) -> Bounds<Pixels> {
        self.resolved_bounds()
    }

    pub fn z_index_value(&self) -> i32 {
        self.z_index
    }

    fn resolved_bounds(&self) -> Bounds<Pixels> {
        let position =
            self.anchor_handle
                .as_ref()
                .map_or(self.position, |handle| AnchoredPosition {
                    bounds: handle.bounds(),
                    ..self.position
                });
        position.resolve(self.size, self.viewport, self.margin)
    }

    fn key(&self) -> OverlayKey {
        OverlayKey {
            id: self.id.clone(),
            bounds: self.resolved_bounds(),
            z_index: self.z_index,
            handle_revision: self.handle.revision(),
            anchor_revision: self
                .anchor_handle
                .as_ref()
                .map_or(0, AnchorHandle::revision),
            has_content: self.content.is_some(),
        }
    }

    fn describe(mut self) -> Description {
        let bounds = self.resolved_bounds();
        let key = self.key();
        let handle = self.handle.clone();
        let dismiss_on_escape = self.dismiss_on_escape;
        let mut interaction = DescriptionInteraction::new(move |event, _window, _app| {
            if dismiss_on_escape
                && matches!(event, InputEvent::KeyDown(event) if event.key.eq_ignore_ascii_case("escape"))
            {
                handle.dismiss();
                return EventResult::HANDLED;
            }
            EventResult::IGNORED
        });
        if let Some(focus_handle) = self.focus_handle {
            interaction = interaction.with_focus(focus_handle);
        }
        let mut description = Description::new::<DeferredOverlay>()
            .id(self.id)
            .diff_key(key)
            .style(absolute_style(bounds))
            .interaction(interaction);
        if let Some(content) = self.content.take() {
            description = description.child(content);
        }
        description
    }
}

impl Element for DeferredOverlay {
    fn into_description(self) -> Description {
        self.describe()
    }
}

/// A retained overlay layer. Higher z values are later in description order,
/// so they paint and hit-test above lower values. Equal z values retain the
/// caller's insertion order.
pub struct OverlayLayer {
    id: ElementId,
    viewport: Rect,
    overlays: Vec<DeferredOverlay>,
}

impl OverlayLayer {
    pub fn new(viewport: Rect) -> Self {
        Self {
            id: ElementId::Name("wgpui-overlay-layer".into()),
            viewport,
            overlays: Vec::new(),
        }
    }

    pub fn id(mut self, id: impl Into<ElementId>) -> Self {
        self.id = id.into();
        self
    }

    pub fn viewport(mut self, viewport: Rect) -> Self {
        self.viewport = viewport;
        self
    }

    pub fn overlay(mut self, overlay: DeferredOverlay) -> Self {
        self.overlays.push(overlay);
        self
    }

    pub fn overlays(mut self, overlays: impl IntoIterator<Item = DeferredOverlay>) -> Self {
        self.overlays.extend(overlays);
        self
    }

    pub fn len(&self) -> usize {
        self.overlays
            .iter()
            .filter(|overlay| overlay.is_renderable())
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn describe(self) -> Description {
        let viewport = self.viewport;
        let mut overlays = self
            .overlays
            .into_iter()
            .map(|mut overlay| {
                if overlay.viewport == Rect::EMPTY {
                    overlay.viewport = viewport;
                }
                overlay
            })
            .filter(DeferredOverlay::is_renderable)
            .collect::<Vec<_>>();
        overlays.sort_by(|left, right| left.z_index.cmp(&right.z_index));
        let key = OverlayLayerKey {
            viewport,
            overlays: overlays.iter().map(DeferredOverlay::key).collect(),
        };
        let mut children = Vec::with_capacity(overlays.len() + 1);
        if let Some(overlay) = overlays
            .iter()
            .rev()
            .find(|overlay| overlay.dismiss_on_click_outside)
        {
            let handle = overlay.handle.clone();
            let interaction = DescriptionInteraction::new(move |event, _window, _app| {
                if matches!(event, InputEvent::Click(_)) {
                    handle.dismiss();
                    EventResult::HANDLED
                } else {
                    EventResult::IGNORED
                }
            });
            children.push(
                Description::new::<OverlayDismissLayer>()
                    .id(ElementId::Name("wgpui-overlay-dismiss".into()))
                    .diff_key(OverlayDismissKey)
                    .style(viewport_style(viewport))
                    .interaction(interaction),
            );
        }
        children.extend(overlays.into_iter().map(DeferredOverlay::describe));
        Description::new::<OverlayLayer>()
            .id(self.id)
            .diff_key(key)
            .style(viewport_style(viewport))
            .children(children)
    }
}

impl Element for OverlayLayer {
    fn into_description(self) -> Description {
        self.describe()
    }
}

fn absolute_style(bounds: Bounds<Pixels>) -> LayoutStyle {
    LayoutStyle {
        position: Position::Absolute,
        inset: wgpui_layout::taffy_tree::LayoutSides {
            left: LengthPercentageAuto::length(bounds.origin.x.value()),
            top: LengthPercentageAuto::length(bounds.origin.y.value()),
            right: LengthPercentageAuto::auto(),
            bottom: LengthPercentageAuto::auto(),
        },
        size: LayoutSize {
            width: Dimension::length(bounds.size.width.value().max(0.0)),
            height: Dimension::length(bounds.size.height.value().max(0.0)),
        },
        ..LayoutStyle::default()
    }
}

fn viewport_style(viewport: Rect) -> LayoutStyle {
    LayoutStyle {
        position: Position::Absolute,
        inset: wgpui_layout::taffy_tree::LayoutSides {
            left: LengthPercentageAuto::length(viewport.min_x),
            top: LengthPercentageAuto::length(viewport.min_y),
            right: LengthPercentageAuto::auto(),
            bottom: LengthPercentageAuto::auto(),
        },
        size: LayoutSize {
            width: Dimension::length(viewport.width().max(0.0)),
            height: Dimension::length(viewport.height().max(0.0)),
        },
        ..LayoutStyle::default()
    }
}

#[derive(Clone, Debug, PartialEq)]
struct OverlayKey {
    id: ElementId,
    bounds: Bounds<Pixels>,
    z_index: i32,
    handle_revision: u64,
    anchor_revision: u64,
    has_content: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct OverlayLayerKey {
    viewport: Rect,
    overlays: Vec<OverlayKey>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OverlayDismissKey;

impl ReconcileKey for OverlayKey {
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
        let Some(previous) = previous.as_any().downcast_ref::<Self>() else {
            return Invalidation::all();
        };
        let mut invalidation = Invalidation::empty();
        if self.bounds != previous.bounds || self.anchor_revision != previous.anchor_revision {
            invalidation |= Invalidation::LAYOUT | Invalidation::HIT;
        }
        if self.z_index != previous.z_index || self.has_content != previous.has_content {
            invalidation |= Invalidation::DISPLAY | Invalidation::HIT;
        }
        if self.id != previous.id || self.handle_revision != previous.handle_revision {
            invalidation = Invalidation::all();
        }
        invalidation
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl ReconcileKey for OverlayLayerKey {
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
        let Some(previous) = previous.as_any().downcast_ref::<Self>() else {
            return Invalidation::all();
        };
        if self == previous {
            return Invalidation::empty();
        }
        let mut invalidation = Invalidation::empty();
        if self.viewport != previous.viewport {
            invalidation |= Invalidation::LAYOUT | Invalidation::HIT;
        }
        if self.overlays != previous.overlays {
            invalidation |= Invalidation::DISPLAY | Invalidation::HIT;
        }
        invalidation
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl ReconcileKey for OverlayDismissKey {
    fn compare(&self, previous: &dyn ReconcileKey) -> Invalidation {
        if previous.as_any().is::<Self>() {
            Invalidation::empty()
        } else {
            Invalidation::all()
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::div::div;
    use crate::styled::Styled;
    use wgpui_core::reconcile::reconciler::Reconciler;
    use wgpui_core::window::{ClickEvent, Window as InteractionWindow};
    use wgpui_layout::taffy_tree::LayoutTree;

    fn viewport() -> Rect {
        Rect::from_origin_size([0.0, 0.0], [400.0, 300.0])
    }

    fn anchor() -> Bounds<Pixels> {
        Bounds::new(
            Point::new(Pixels(100.0), Pixels(80.0)),
            Size::pixels(40.0, 20.0),
        )
    }

    fn overlay(id: u64, z_index: i32) -> DeferredOverlay {
        DeferredOverlay::new(id, div().w(40.0).h(30.0))
            .anchor(anchor(), Anchor::BottomLeft)
            .viewport(viewport())
            .size(Size::pixels(40.0, 30.0))
            .z_index(z_index)
    }

    #[test]
    fn closed_and_empty_entries_do_not_create_overlay_content() {
        assert!(
            OverlayLayer::new(viewport())
                .overlay(DeferredOverlay::empty(2u64))
                .is_empty()
        );
        let handle = OverlayHandle::new();
        handle.close();
        let layer = OverlayLayer::new(viewport())
            .overlay(overlay(1, 10))
            .overlay(DeferredOverlay::empty(2u64).controller(handle));
        assert_eq!(layer.len(), 1);
        assert!(!layer.is_empty());
    }

    #[test]
    fn outside_click_routes_to_the_topmost_dismissible_overlay() {
        let handle = OverlayHandle::new();
        let description = Element::into_description(
            OverlayLayer::new(viewport()).overlay(
                overlay(1, 1)
                    .controller(handle.clone())
                    .dismiss_on_click_outside(true),
            ),
        );
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let mut plan = match reconciler.reconcile(description, &mut layout) {
            Ok(plan) => plan,
            Err(error) => panic!("overlay reconciliation failed: {error}"),
        };
        let interaction_count = plan.nodes().len();
        let mut interaction =
            match (0..interaction_count).find_map(|index| plan.take_interaction(index)) {
                Some(interaction) => interaction,
                None => panic!("dismiss layer interaction was not retained"),
            };
        let mut window = InteractionWindow::new();
        let mut app = wgpui_core::app::App::create();
        assert_eq!(
            interaction.dispatch(
                &InputEvent::Click(ClickEvent::default()),
                &mut window,
                &mut app,
            ),
            EventResult::HANDLED
        );
        assert!(!handle.is_open());
    }

    #[test]
    fn stacking_order_is_retained_in_the_description_tree() {
        let description = Element::into_description(
            OverlayLayer::new(viewport())
                .overlay(overlay(1, 20))
                .overlay(overlay(2, 10)),
        );
        let children = description.child_descriptions();
        assert_eq!(children.len(), 2);
        assert!(matches!(
            children[0].element_id(),
            Some(ElementId::Integer(2))
        ));
        assert!(matches!(
            children[1].element_id(),
            Some(ElementId::Integer(1))
        ));
    }

    #[test]
    fn focus_handle_is_attached_to_the_retained_overlay_node() {
        let focus_handle = FocusHandle::new();
        let description = Element::into_description(overlay(1, 1).focus_handle(focus_handle));
        let mut reconciler = Reconciler::new();
        let mut layout = LayoutTree::new();
        let mut plan = match reconciler.reconcile(description, &mut layout) {
            Ok(plan) => plan,
            Err(error) => panic!("overlay reconciliation failed: {error}"),
        };
        let interaction = match plan.take_interaction(0) {
            Some(interaction) => interaction,
            None => panic!("overlay descriptions always have interaction metadata"),
        };
        assert_eq!(interaction.focus_handle(), Some(focus_handle));
    }

    #[test]
    fn controller_teardown_closes_the_retained_entry_safely() {
        let handle = OverlayHandle::new();
        let retained_handle = handle.clone();
        drop(handle);
        retained_handle.close();
        let layer =
            OverlayLayer::new(viewport()).overlay(overlay(1, 1).controller(retained_handle));
        assert!(layer.is_empty());
    }

    #[test]
    fn dismissal_is_idempotent_and_notifies_the_native_boundary_once() {
        let notifications = Rc::new(RefCell::new(0));
        let handle = OverlayHandle::new();
        let count = notifications.clone();
        handle.invalidate_with(move || *count.borrow_mut() += 1);
        let dismissed = Rc::new(RefCell::new(0));
        let count = dismissed.clone();
        handle.on_dismiss(move || *count.borrow_mut() += 1);
        assert!(handle.dismiss());
        assert!(!handle.dismiss());
        assert_eq!(*notifications.borrow(), 1);
        assert_eq!(*dismissed.borrow(), 1);
    }

    #[test]
    fn overlay_keys_split_geometry_from_stacking_changes() {
        let left = overlay(1, 1);
        let moved = overlay(1, 1).offset(Point::new(Pixels(4.0), Pixels(0.0)));
        assert!(
            left.key()
                .compare(&moved.key())
                .contains(Invalidation::LAYOUT)
        );
        let raised = overlay(1, 2);
        assert!(
            left.key()
                .compare(&raised.key())
                .contains(Invalidation::DISPLAY)
        );
    }
}
