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

use crate::app::App;
use crate::boundary::compositor::ExternalSurfaceId;
use crate::boundary::policy::BoundaryPolicy;
use crate::patch::emit::Emit;
use crate::reconcile::diff_key::ReconcileKey;
use crate::action::Action;
use crate::window::{DragData, EventResult, FocusHandle, InputEvent, Window};
use std::any::TypeId;
use std::sync::Arc;
use wgpui_layout::taffy_tree::{LayoutRect, LayoutStyle};

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

/// Renderer-neutral text properties inherited by raw string children.
///
/// The core description must not depend on a font or glyph atlas, but it does
/// need to carry the resolved style far enough for the renderer to measure raw
/// text before laying out the tree. `None` means that the renderer should use
/// the inherited value or its platform default.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextOptions {
    pub size: Option<f32>,
    pub line_height: Option<f32>,
    pub color: Option<[f32; 4]>,
    pub weight: Option<u16>,
    pub italic: Option<bool>,
    pub alignment: Option<u8>,
    pub nowrap: Option<bool>,
    pub ellipsis: Option<bool>,
    pub line_clamp: Option<usize>,
    pub letter_spacing: Option<f32>,
    pub underline: Option<TextDecoration>,
    pub strikethrough: Option<TextDecoration>,
    pub gradient: Option<Vec<([f32; 4], f32)>>,
    pub gradient_angle: Option<f32>,
}

/// A renderer-neutral text decoration.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TextDecoration {
    pub thickness: f32,
    pub color: Option<[f32; 4]>,
    pub wavy: bool,
}

/// Metadata for a texture produced outside the retained scene.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ExternalSurfaceProperties {
    /// The registry identity of the producer-owned texture.
    pub id: ExternalSurfaceId,
    /// Straight alpha applied while compositing the texture.
    pub opacity: f32,
    /// Uniform corner radius applied while compositing the texture.
    pub corner_radius: f32,
}

/// Input callbacks carried from an element description to the native window.
/// The callback is deliberately stored beside the description until layout has
/// produced the element's actual bounds; registering it earlier would make
/// hit testing use stale geometry after a resize or a retained relayout.
type InteractionCallback = Box<dyn FnMut(&InputEvent, &mut Window, &mut App) -> EventResult>;
type ActionCallback = Box<dyn FnMut(&dyn Action, &mut Window, &mut App) -> EventResult>;
type DragStartCallback = Box<dyn FnMut(&DragData, &mut Window, &mut App)>;
type DragHoverCallback = Box<dyn FnMut(bool, &DragData, &mut Window, &mut App) -> EventResult>;
type DropCallback = Box<dyn FnMut(&DragData, &mut Window, &mut App) -> EventResult>;
enum LayoutCallback {
    Bounds(Box<dyn FnMut(LayoutRect)>),
    BoundsChanged(Box<dyn FnMut(LayoutRect) -> bool>),
    Content(Box<dyn FnMut(LayoutRect, LayoutRect)>),
    ContentChanged(Box<dyn FnMut(LayoutRect, LayoutRect) -> bool>),
}

pub struct DescriptionInteraction {
    callback: InteractionCallback,
    action_callback: Option<ActionCallback>,
    focus_handle: Option<FocusHandle>,
    drag_source: Option<DragData>,
    drag_start_callback: Option<DragStartCallback>,
    drag_hover_callback: Option<DragHoverCallback>,
    drop_callback: Option<DropCallback>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ScrollInfo {
    pub handle_id: u64,
    pub content_size: [f32; 2],
    pub max_offset: [f32; 2],
    pub offset: [f32; 2],
}

pub struct DescriptionLayout {
    callback: LayoutCallback,
}

impl std::fmt::Debug for DescriptionLayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("DescriptionLayout(..)")
    }
}

impl DescriptionLayout {
    pub fn new(callback: impl FnMut(LayoutRect) + 'static) -> Self {
        Self {
            callback: LayoutCallback::Bounds(Box::new(callback)),
        }
    }

    pub fn apply(&mut self, bounds: LayoutRect) -> bool {
        match &mut self.callback {
            LayoutCallback::Bounds(callback) => {
                callback(bounds);
                false
            }
            LayoutCallback::BoundsChanged(callback) => callback(bounds),
            LayoutCallback::Content(callback) => {
                callback(bounds, bounds);
                false
            }
            LayoutCallback::ContentChanged(callback) => callback(bounds, bounds),
        }
    }

    pub fn new_changed(callback: impl FnMut(LayoutRect) -> bool + 'static) -> Self {
        Self {
            callback: LayoutCallback::BoundsChanged(Box::new(callback)),
        }
    }

    pub fn with_content(callback: impl FnMut(LayoutRect, LayoutRect) + 'static) -> Self {
        Self {
            callback: LayoutCallback::Content(Box::new(callback)),
        }
    }

    pub fn apply_with_content(&mut self, bounds: LayoutRect, content: LayoutRect) -> bool {
        match &mut self.callback {
            LayoutCallback::Bounds(callback) => {
                callback(bounds);
                false
            }
            LayoutCallback::BoundsChanged(callback) => callback(bounds),
            LayoutCallback::Content(callback) => {
                callback(bounds, content);
                false
            }
            LayoutCallback::ContentChanged(callback) => callback(bounds, content),
        }
    }

    pub fn with_content_changed(
        callback: impl FnMut(LayoutRect, LayoutRect) -> bool + 'static,
    ) -> Self {
        Self {
            callback: LayoutCallback::ContentChanged(Box::new(callback)),
        }
    }
}

impl std::fmt::Debug for DescriptionInteraction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("DescriptionInteraction(..)")
    }
}

impl DescriptionInteraction {
    pub fn new(
        callback: impl FnMut(&InputEvent, &mut Window, &mut App) -> EventResult + 'static,
    ) -> Self {
        Self {
            callback: Box::new(callback),
            action_callback: None,
            focus_handle: None,
            drag_source: None,
            drag_start_callback: None,
            drag_hover_callback: None,
            drop_callback: None,
        }
    }

    pub fn dispatch(
        &mut self,
        event: &InputEvent,
        window: &mut Window,
        app: &mut App,
    ) -> EventResult {
        (self.callback)(event, window, app)
    }

    pub fn with_focus(mut self, focus_handle: FocusHandle) -> Self {
        self.focus_handle = Some(focus_handle);
        self
    }

    pub fn focus_handle(&self) -> Option<FocusHandle> {
        self.focus_handle
    }

    pub fn with_drag_source(
        mut self,
        data: DragData,
        callback: impl FnMut(&DragData, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.drag_source = Some(data);
        self.drag_start_callback = Some(Box::new(callback));
        self
    }

    pub fn drag_source(&self) -> Option<DragData> {
        self.drag_source.clone()
    }

    pub fn start_drag(&mut self, data: &DragData, window: &mut Window, app: &mut App) {
        if let Some(callback) = self.drag_start_callback.as_deref_mut() {
            callback(data, window, app);
        }
    }

    pub fn with_drag_hover_handler(
        mut self,
        callback: impl FnMut(bool, &DragData, &mut Window, &mut App) -> EventResult + 'static,
    ) -> Self {
        self.drag_hover_callback = Some(Box::new(callback));
        self
    }

    pub fn dispatch_drag_hover(
        &mut self,
        hovered: bool,
        data: &DragData,
        window: &mut Window,
        app: &mut App,
    ) -> EventResult {
        self.drag_hover_callback.as_deref_mut().map_or(EventResult::IGNORED, |callback| {
            callback(hovered, data, window, app)
        })
    }

    pub fn with_drop_handler(
        mut self,
        callback: impl FnMut(&DragData, &mut Window, &mut App) -> EventResult + 'static,
    ) -> Self {
        self.drop_callback = Some(Box::new(callback));
        self
    }

    pub fn dispatch_drop(
        &mut self,
        data: &DragData,
        window: &mut Window,
        app: &mut App,
    ) -> EventResult {
        self.drop_callback.as_deref_mut().map_or(EventResult::IGNORED, |callback| {
            callback(data, window, app)
        })
    }

    pub fn on_action<A: Action>(
        mut self,
        mut handler: impl FnMut(&A, &mut Window, &mut App) -> EventResult + 'static,
    ) -> Self {
        let mut previous = self.action_callback.take();
        self.action_callback = Some(Box::new(move |action, window, app| {
            let mut result = previous
                .as_deref_mut()
                .map_or(EventResult::IGNORED, |callback| callback(action, window, app));
            if !result.propagate {
                return result;
            }
            if let Some(action) = action.as_any().downcast_ref::<A>() {
                let current = handler(action, window, app);
                if current.handled {
                    result = current;
                } else {
                    result.propagate |= current.propagate;
                }
            }
            result
        }));
        self
    }

    pub fn with_action_handler(
        mut self,
        mut handler: impl FnMut(&dyn Action, &mut Window, &mut App) -> EventResult + 'static,
    ) -> Self {
        let mut previous = self.action_callback.take();
        self.action_callback = Some(Box::new(move |action, window, app| {
            let previous_result = previous
                .as_deref_mut()
                .map_or(EventResult::IGNORED, |callback| callback(action, window, app));
            if !previous_result.propagate {
                return previous_result;
            }
            let current_result = handler(action, window, app);
            if current_result.handled {
                current_result
            } else {
                EventResult {
                    handled: previous_result.handled,
                    propagate: previous_result.propagate || current_result.propagate,
                }
            }
        }));
        self
    }

    pub fn dispatch_action(
        &mut self,
        action: &dyn Action,
        window: &mut Window,
        app: &mut App,
    ) -> EventResult {
        self.action_callback
            .as_deref_mut()
            .map_or(EventResult::IGNORED, |callback| callback(action, window, app))
    }

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
    pub(crate) scroll_axes: [bool; 2],
    pub(crate) automatic_scroll: bool,
    pub(crate) emitter: Option<Box<dyn Emit>>,
    pub(crate) layout_style: LayoutStyle,
    pub(crate) children: Vec<Description>,
    pub(crate) clip_children: bool,
    pub(crate) raw_text: Option<RawText>,
    pub(crate) text_size: Option<f32>,
    pub(crate) text_color: Option<[f32; 4]>,
    pub(crate) text_options: TextOptions,
    pub(crate) interaction: Option<DescriptionInteraction>,
    pub(crate) scroll_info: Option<ScrollInfo>,
    pub(crate) layout_callback: Option<DescriptionLayout>,
    pub(crate) external_surface: Option<ExternalSurfaceProperties>,
    pub(crate) active_animation: bool,
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
            .field("scroll_axes", &self.scroll_axes)
            .field("automatic_scroll", &self.automatic_scroll)
            .field("has_emitter", &self.emitter.is_some())
            .field("children", &self.children.len())
            .field("clip_children", &self.clip_children)
            .field("has_layout_callback", &self.layout_callback.is_some())
            .field("active_animation", &self.active_animation)
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
            scroll_axes: [false, false],
            automatic_scroll: false,
            emitter: None,
            layout_style: LayoutStyle::default(),
            children: Vec::new(),
            clip_children: false,
            raw_text: None,
            text_size: None,
            text_color: None,
            text_options: TextOptions::default(),
            interaction: None,
            scroll_info: None,
            layout_callback: None,
            external_surface: None,
            active_animation: false,
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

    pub fn scroll_info(mut self, scroll_info: ScrollInfo) -> Self {
        self.scroll_info = Some(scroll_info);
        self
    }

    /// Declare which overflow axes may establish an automatic scroll root.
    pub fn with_scroll_axes(mut self, axes: [bool; 2]) -> Self {
        self.scroll_axes = axes;
        self
    }

    /// Allow the native backend to retain and route scroll input for this
    /// element when its configured overflow has a scrollable extent.
    pub fn automatic_scroll(mut self, automatic: bool) -> Self {
        self.automatic_scroll = automatic;
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
        self.layout_style.flex_shrink = 0.0;
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

    /// Carry the resolved inherited text metrics to a renderer-owned text
    /// materializer without coupling the core description to a font system.
    pub fn text_metrics(mut self, size: Option<f32>, color: Option<[f32; 4]>) -> Self {
        self.text_size = size;
        self.text_color = color;
        self
    }

    pub fn text_metrics_value(&self) -> (Option<f32>, Option<[f32; 4]>) {
        (self.text_size, self.text_color)
    }

    /// Carry the complete resolved text style to a renderer-owned materializer.
    pub fn text_options(mut self, options: TextOptions) -> Self {
        self.text_options = options;
        self
    }

    /// Read the local text style during renderer materialization.
    pub fn text_options_value(&self) -> &TextOptions {
        &self.text_options
    }

    pub fn interaction(mut self, interaction: DescriptionInteraction) -> Self {
        self.interaction = Some(interaction);
        self
    }

    pub fn on_layout(mut self, callback: impl FnMut(LayoutRect) + 'static) -> Self {
        self.layout_callback = Some(DescriptionLayout::new(callback));
        self
    }

    pub fn on_layout_with_content(
        mut self,
        callback: impl FnMut(LayoutRect, LayoutRect) + 'static,
    ) -> Self {
        self.layout_callback = Some(DescriptionLayout::with_content(callback));
        self
    }

    pub fn on_layout_changed(mut self, callback: impl FnMut(LayoutRect) -> bool + 'static) -> Self {
        self.layout_callback = Some(DescriptionLayout::new_changed(callback));
        self
    }

    pub fn on_layout_with_content_changed(
        mut self,
        callback: impl FnMut(LayoutRect, LayoutRect) -> bool + 'static,
    ) -> Self {
        self.layout_callback = Some(DescriptionLayout::with_content_changed(callback));
        self
    }

    /// Make this leaf sample a texture produced by an external renderer.
    ///
    /// The texture is not copied into the scene and its pixels are not part of
    /// reconciliation. The emitter turns the resolved geometry into one
    /// compositor entry, so a producer update only damages that entry's
    /// visible rectangle.
    pub fn external_surface(
        mut self,
        id: ExternalSurfaceId,
        opacity: f32,
        corner_radius: f32,
    ) -> Self {
        self.external_surface = Some(ExternalSurfaceProperties {
            id,
            opacity,
            corner_radius,
        });
        self
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

    /// The external texture metadata, if this description is a surface leaf.
    pub fn external_surface_properties(&self) -> Option<ExternalSurfaceProperties> {
        self.external_surface
    }

    /// Mark this description as needing another frame while an animation is
    /// active. This metadata does not participate in reconciliation; the
    /// sampled style or primitive remains the source of display invalidation.
    pub fn active_animation(mut self) -> Self {
        self.active_animation = true;
        self
    }

    /// Whether this description or one of its descendants is still animating.
    pub fn has_active_animation(&self) -> bool {
        self.active_animation
            || self
                .children
                .iter()
                .any(Description::has_active_animation)
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

    pub fn scroll_axes(&self) -> [bool; 2] {
        self.scroll_axes
    }

    pub fn has_automatic_scroll(&self) -> bool {
        self.automatic_scroll
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

    #[test]
    fn intrinsic_text_size_cannot_be_flex_shrunk() {
        let mut description = Description::new::<Panel>().style(LayoutStyle {
            flex_shrink: 1.0,
            ..LayoutStyle::default()
        });

        description.set_intrinsic_size(120.0, 20.0);

        assert_eq!(description.layout_style.flex_shrink, 0.0);
        assert_eq!(
            description.layout_style.size.width,
            wgpui_layout::taffy_tree::Dimension::length(120.0)
        );
        assert_eq!(
            description.layout_style.size.height,
            wgpui_layout::taffy_tree::Dimension::length(20.0)
        );
    }
}
