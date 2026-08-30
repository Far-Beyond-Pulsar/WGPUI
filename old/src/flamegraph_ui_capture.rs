//! On-demand UI-tree capture (Phase 5 of the profiling epic, see issue #61):
//! the CPU/UI-framework-specific half of "RenderDoc for our UI framework".
//! Where Phase 4's `DeepCapture` (`flamegraph.rs`) answers "what did the GPU
//! actually do for this frame," this module answers "why did the tree look
//! the way it did" -- element tree shape, computed layout, resolved styles,
//! and the resulting paint primitive list, for one triggered frame.
//!
//! # Design: independent of `DeepCapture`, same shape
//!
//! Like Phase 4, this is a triggered, single-frame, opt-in capture -- element
//! tree + layout + Scene data for every frame, all the time, would be far too
//! much data to ring-buffer alongside Phase 1's lightweight spans. The public
//! arm/retrieve API deliberately mirrors `DeepCapture`'s shape
//! (`request_*`/`take_*_request`/`complete_*`/`take_completed_*`, one
//! "requested" `AtomicBool` plus one "last completed result" mailbox) for
//! consistency and because it is a proven, working pattern for this exact
//! problem shape.
//!
//! It is deliberately a *separate*, independently-triggerable mechanism
//! rather than sharing `DeepCapture`'s state machine, for two reasons:
//!
//! 1. **Different trigger points.** `DeepCapture` fires inside
//!    `WgpuRenderer::draw()` (the `platform::cross` renderer layer) and needs
//!    multi-frame-latency async GPU buffer readback (`flamegraph_gpu.rs`'s
//!    `map_async`/poll-on-render-thread dance). This capture fires inside
//!    `Window::draw`/`draw_roots` (the windowing/element layer, upstream of
//!    the renderer entirely) and completes **synchronously** within that one
//!    call -- there is no GPU readback here at all, so there is no "pending"
//!    state to model, no polling, no teardown-after-harvest. Coupling two
//!    state machines with such different completion semantics would only
//!    make both harder to reason about.
//! 2. **Independent usefulness.** A caller debugging "why is this element
//!    this color" doesn't need a GPU command stream, and a caller debugging
//!    "what did draw call N actually read" doesn't need element/style data.
//!    Keeping them separate lets either be requested alone, while still
//!    allowing both to be requested together for the same triggered frame
//!    (each arms independently and fires on its own next natural hook point).
//!
//! # Where this hooks in
//!
//! - **Element tree structure + computed layout**: `Drawable::prepaint`
//!   (`element.rs`) already resolves `Bounds<Pixels>` for every element via
//!   `window.layout_bounds(layout_id)` before descending into children --
//!   exactly Taffy's resolved box for that node. Recording a
//!   [`UiElementNode`] there, in the same push-before-descend order the CPU
//!   span stack (`flamegraph.rs`) already uses for `depth`, gives a flat
//!   DFS-ordered vector that reconstructs the full tree shape with no parent
//!   pointers needed.
//! - **Resolved style**: only `Interactivity`-backed (`Div`-derived)
//!   elements have a generic notion of "resolved `Style`" -- arbitrary
//!   `Element` impls (e.g. `Canvas`) paint however they like and have no
//!   style to resolve. `Interactivity::paint` (`elements/div.rs`) already
//!   computes the final, hover/focus/drag-inclusive `Style` for its element;
//!   this module records a lightweight [`UiStyleSnapshot`] there, keyed by
//!   the same `GlobalElementId` hash `ElementAttribution` already uses, and
//!   joins it onto the matching tree node when the capture is finalized.
//!   Elements with no stable id (hash `0`, the same "no id" sentinel
//!   `ElementAttribution::global_id_hash` uses) are not join targets, since a
//!   hash of `0` is not a unique key.
//! - **Scene snapshot**: `Window::draw` finishes and sorts `next_frame.scene`
//!   before swapping it into `rendered_frame`; that is the exact "paint
//!   primitive list handed to the renderer" the issue asks for, deep-copied
//!   here instead of being overwritten by the next frame.
//!
//! # Scope note: Scene primitive fidelity
//!
//! Every `scene.rs` primitive type (`Quad`, `Shadow`, ...) already derives
//! `Clone` (used throughout `Scene::insert_primitive`/`replay`), so the
//! "deep-copy, don't reinvent" path the issue calls out is available. They
//! are `pub(crate)` types, though, and embedding a `pub(crate)` type in this
//! module's `pub` API would fail to compile (a private type leaking into a
//! public interface) -- so rather than widen `scene.rs`'s internal type
//! visibility (a bigger, riskier change touching GPU-layout-sensitive
//! `#[repr(C)]` structs), each primitive is copied field-by-field into a
//! small `pub` snapshot type defined here, keeping the essentials (bounds,
//! resolved color, atlas identity) and leaving out geometry that doesn't
//! serve the "why does this look the way it does" debugging goal for this
//! first cut (per-primitive `content_mask`, `corner_radii`/`border_widths`
//! minutiae, and full path vertex lists). `FilterBoundary` markers are
//! skipped entirely (bookkeeping, not visible content) -- the same primitive
//! set Phase 4's `DrawCallKind` covers.

use std::{
    any::type_name,
    cell::RefCell,
    collections::HashMap,
    sync::atomic::{AtomicBool, Ordering},
};

use serde::Serialize;

use crate::{Background, Bounds, Element, Fill, GlobalElementId, Hsla, Pixels, ScaledPixels, Scene, Style, TextColor};

/// Logical-pixel bounds, decoupled from the crate's internal `Bounds<Pixels>`
/// representation so this module's public types don't need to reason about
/// `Bounds<T>`'s trait bounds. Origin is window-relative (top-left), matching
/// [`crate::Window::layout_bounds`]'s convention.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct UiBounds {
    /// Left edge, in logical pixels from the window's left edge.
    pub x: f32,
    /// Top edge, in logical pixels from the window's top edge.
    pub y: f32,
    /// Width, in logical pixels.
    pub width: f32,
    /// Height, in logical pixels.
    pub height: f32,
}

fn logical_bounds(bounds: Bounds<Pixels>) -> UiBounds {
    UiBounds {
        x: f32::from(bounds.origin.x),
        y: f32::from(bounds.origin.y),
        width: f32::from(bounds.size.width),
        height: f32::from(bounds.size.height),
    }
}

/// Converts scaled (device-density-adjusted) pixel bounds, as scene
/// primitives store them, back to logical pixels using the window's scale
/// factor -- the same units [`UiElementNode::bounds`] uses, so a viewer can
/// compare element bounds and scene primitive bounds directly.
fn scaled_bounds_to_logical(bounds: Bounds<ScaledPixels>, scale_factor: f32) -> UiBounds {
    let scale = if scale_factor > 0.0 { scale_factor } else { 1.0 };
    UiBounds {
        x: bounds.origin.x.0 / scale,
        y: bounds.origin.y.0 / scale,
        width: bounds.size.width.0 / scale,
        height: bounds.size.height.0 / scale,
    }
}

/// Resolved style values for one element at paint time (Phase 5, issue #61),
/// captured only for `Interactivity`-backed (`Div`-derived) elements -- see
/// this module's doc comment for why not every `Element` has a generic style
/// to resolve. Intentionally a curated subset of `Style`'s many fields, not
/// the whole struct: the fields here are exactly the ones that answer "why
/// does this element look the way it does" (visibility/position/opacity for
/// "is it even shown," background/border/text color for "why this color").
/// Layout-affecting fields (flex/grid/sizing) are deliberately left out --
/// [`UiElementNode::bounds`] already carries Taffy's *resolved* answer to
/// those, which is what a debugging session actually wants to compare
/// against, rather than the pre-layout inputs.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct UiStyleSnapshot {
    /// `Style::display`.
    pub display: crate::Display,
    /// `Style::visibility`.
    pub visibility: crate::Visibility,
    /// `Style::position`.
    pub position: crate::Position,
    /// `Style::opacity`.
    pub opacity: Option<f32>,
    /// `Style::background`.
    pub background: Option<Fill>,
    /// `Style::border_color`.
    pub border_color: Option<Hsla>,
    /// `Style::text`'s resolved text color, when the element's text style
    /// refinement sets one.
    pub text_color: Option<TextColor>,
}

impl UiStyleSnapshot {
    fn from_style(style: &Style) -> Self {
        Self {
            display: style.display,
            visibility: style.visibility,
            position: style.position,
            opacity: style.opacity,
            background: style.background.clone(),
            border_color: style.border_color,
            text_color: style.text.color,
        }
    }
}

/// One node in a captured element tree, in prepaint DFS order. Mirrors
/// `CpuSpan::depth`'s flat-vector-plus-depth trick (`flamegraph.rs`): a
/// viewer reconstructs the tree from `depth` alone, without parent pointers.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UiElementNode {
    /// `type_name::<E>()` of the concrete `Element` implementation, same
    /// convention as `ElementAttribution::type_name` (Phase 1).
    pub type_name: &'static str,
    /// A `seahash` hash of the element's `GlobalElementId`, computed with the
    /// exact same function Phase 1's `ElementAttribution::global_id_hash`
    /// uses (`hash_global_element_id`, `flamegraph.rs`) so the two are
    /// directly comparable. Zero when the element has no stable id.
    pub global_id_hash: u64,
    /// Nesting depth, taken from the capture's traversal-depth counter at
    /// prepaint time -- the same "depth at push time" convention `CpuSpan`
    /// uses, independent of whether the element has a stable id (unlike the
    /// inspector's `GlobalElementId`-length-based depth, which only advances
    /// for elements that push an id).
    pub depth: u16,
    /// Taffy's resolved box for this element (`window.layout_bounds`), in
    /// logical pixels, window-relative.
    pub bounds: UiBounds,
    /// Resolved style, when this element is `Interactivity`-backed (`Div`) and
    /// has a stable id. `None` for elements with no generic style to resolve,
    /// or with no stable id to join a captured style onto.
    pub style: Option<UiStyleSnapshot>,
}

/// One captured `Quad` primitive.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct UiQuadSnapshot {
    /// Logical-pixel bounds.
    pub bounds: UiBounds,
    /// Resolved background fill.
    pub background: Background,
    /// Border color (alpha `0` when the quad has no visible border).
    pub border_color: Hsla,
}

/// One captured `Shadow` primitive.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct UiShadowSnapshot {
    /// Logical-pixel bounds.
    pub bounds: UiBounds,
    /// Shadow color.
    pub color: Hsla,
    /// Blur radius, in logical pixels.
    pub blur_radius: f32,
}

/// One captured `Underline` primitive.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct UiUnderlineSnapshot {
    /// Logical-pixel bounds.
    pub bounds: UiBounds,
    /// Underline color.
    pub color: Hsla,
}

/// One captured `Path` primitive. Bounds and fill color only -- vertex
/// geometry is left out this round, see this module's doc comment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct UiPathSnapshot {
    /// Logical-pixel bounds.
    pub bounds: UiBounds,
    /// Fill color/gradient.
    pub color: Background,
}

/// One captured `MonochromeSprite` primitive (glyphs, monochrome icons).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct UiMonochromeSpriteSnapshot {
    /// Logical-pixel bounds.
    pub bounds: UiBounds,
    /// Text/tint color.
    pub color: TextColor,
    /// `(AtlasTextureKind as u64) << 32 | AtlasTextureId::index`, matching
    /// the encoding Phase 4's `DeepCaptureDrawCall::atlas_texture_id` uses.
    pub atlas_texture_id: u64,
}

/// One captured `PolychromeSprite` primitive (images).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct UiPolychromeSpriteSnapshot {
    /// Logical-pixel bounds.
    pub bounds: UiBounds,
    /// Element opacity applied to this sprite.
    pub opacity: f32,
    /// Same atlas-id encoding as [`UiMonochromeSpriteSnapshot::atlas_texture_id`].
    pub atlas_texture_id: u64,
}

/// One captured `BackdropFilter` primitive.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct UiBackdropFilterSnapshot {
    /// Logical-pixel bounds.
    pub bounds: UiBounds,
    /// Blur radius, in logical pixels.
    pub blur_radius: f32,
    /// Element opacity captured at paint time.
    pub opacity: f32,
}

/// One captured `PaintSurface` primitive (embedded WGPU/video surfaces).
/// Bounds only -- the surface's backing content is `SurfaceRegistry` state
/// that Phase 4b (#72) is the natural place to capture, not this phase.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct UiSurfaceSnapshot {
    /// Logical-pixel bounds.
    pub bounds: UiBounds,
}

/// A deep copy of the triggered frame's finished `Scene` -- the actual paint
/// primitive list handed to the renderer, per primitive kind. See this
/// module's doc comment for the scope note on why these are lightweight
/// per-primitive snapshots rather than the internal `scene.rs` types
/// directly.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct SceneSnapshot {
    /// Captured `Quad` primitives, in the scene's paint order.
    pub quads: Vec<UiQuadSnapshot>,
    /// Captured `Shadow` primitives.
    pub shadows: Vec<UiShadowSnapshot>,
    /// Captured `Underline` primitives.
    pub underlines: Vec<UiUnderlineSnapshot>,
    /// Captured `Path` primitives.
    pub paths: Vec<UiPathSnapshot>,
    /// Captured `MonochromeSprite` primitives.
    pub monochrome_sprites: Vec<UiMonochromeSpriteSnapshot>,
    /// Captured `PolychromeSprite` primitives.
    pub polychrome_sprites: Vec<UiPolychromeSpriteSnapshot>,
    /// Captured `BackdropFilter` primitives.
    pub backdrop_filters: Vec<UiBackdropFilterSnapshot>,
    /// Captured `PaintSurface` primitives.
    pub surfaces: Vec<UiSurfaceSnapshot>,
}

impl SceneSnapshot {
    /// Total primitive count across every kind, for a quick "how much did
    /// this frame paint" headline number.
    pub fn primitive_count(&self) -> usize {
        self.quads.len()
            + self.shadows.len()
            + self.underlines.len()
            + self.paths.len()
            + self.monochrome_sprites.len()
            + self.polychrome_sprites.len()
            + self.backdrop_filters.len()
            + self.surfaces.len()
    }

    fn capture(scene: &Scene, scale_factor: f32) -> Self {
        let quads = scene
            .quads
            .iter()
            .map(|quad| UiQuadSnapshot {
                bounds: scaled_bounds_to_logical(quad.bounds, scale_factor),
                background: quad.background,
                border_color: quad.border_color,
            })
            .collect();

        let shadows = scene
            .shadows
            .iter()
            .map(|shadow| UiShadowSnapshot {
                bounds: scaled_bounds_to_logical(shadow.bounds, scale_factor),
                color: shadow.color,
                blur_radius: shadow.blur_radius.0 / scale_factor.max(1.0).max(f32::MIN_POSITIVE),
            })
            .collect();

        let underlines = scene
            .underlines
            .iter()
            .map(|underline| UiUnderlineSnapshot {
                bounds: scaled_bounds_to_logical(underline.bounds, scale_factor),
                color: underline.color,
            })
            .collect();

        let paths = scene
            .paths
            .iter()
            .map(|path| UiPathSnapshot {
                bounds: scaled_bounds_to_logical(path.bounds, scale_factor),
                color: path.color,
            })
            .collect();

        let monochrome_sprites = scene
            .monochrome_sprites
            .iter()
            .map(|sprite| UiMonochromeSpriteSnapshot {
                bounds: scaled_bounds_to_logical(sprite.bounds, scale_factor),
                color: sprite.text_color,
                atlas_texture_id: encode_atlas_texture_id(
                    sprite.tile.texture_id.kind as u64,
                    sprite.tile.texture_id.index,
                ),
            })
            .collect();

        let polychrome_sprites = scene
            .polychrome_sprites
            .iter()
            .map(|sprite| UiPolychromeSpriteSnapshot {
                bounds: scaled_bounds_to_logical(sprite.bounds, scale_factor),
                opacity: sprite.opacity,
                atlas_texture_id: encode_atlas_texture_id(
                    sprite.tile.texture_id.kind as u64,
                    sprite.tile.texture_id.index,
                ),
            })
            .collect();

        let backdrop_filters = scene
            .backdrop_filters
            .iter()
            .map(|filter| UiBackdropFilterSnapshot {
                bounds: scaled_bounds_to_logical(filter.bounds, scale_factor),
                blur_radius: filter.blur_radius.0 / scale_factor.max(f32::MIN_POSITIVE),
                opacity: filter.opacity,
            })
            .collect();

        let surfaces = scene
            .surfaces
            .iter()
            .map(|surface| UiSurfaceSnapshot {
                bounds: scaled_bounds_to_logical(surface.bounds, scale_factor),
            })
            .collect();

        Self {
            quads,
            shadows,
            underlines,
            paths,
            monochrome_sprites,
            polychrome_sprites,
            backdrop_filters,
            surfaces,
        }
    }
}

fn encode_atlas_texture_id(kind: u64, index: u32) -> u64 {
    (kind << 32) | (index as u64)
}

/// One on-demand, single-frame UI-tree capture (Phase 5 of the profiling
/// epic, issue #61): element tree shape, computed layout, resolved style,
/// and the paint primitive list, for exactly one triggered `Window::draw()`
/// call. See this module's doc comment for the full design rationale.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct UiTreeCapture {
    /// Hashed `WindowId` this capture belongs to, same convention as
    /// `FrameCapture::window_id` (Phase 1).
    pub window_id: u64,
    /// Every element that went through `Drawable::prepaint` during the
    /// captured frame, in DFS order. See [`UiElementNode::depth`] for how to
    /// reconstruct the tree shape.
    pub nodes: Vec<UiElementNode>,
    /// The frame's finished `Scene`, deep-copied at the point `Window::draw`
    /// would otherwise let the next frame overwrite it.
    pub scene: SceneSnapshot,
}

impl UiTreeCapture {
    /// The node for `global_id_hash`, if any element in this capture's tree
    /// has that hash and it is not the "no stable id" sentinel (`0`).
    pub fn node_by_global_id_hash(&self, global_id_hash: u64) -> Option<&UiElementNode> {
        if global_id_hash == 0 {
            return None;
        }
        self.nodes
            .iter()
            .find(|node| node.global_id_hash == global_id_hash)
    }
}

/// Whether a UI-tree capture has been armed (via
/// [`request_ui_tree_capture`]) but not yet consumed by a `Window::draw()`
/// call.
static UI_TREE_CAPTURE_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Whether a UI-tree capture is actively recording on the calling thread
/// right now (between `maybe_begin_capture` and `finish_capture`/an
/// [`ActiveUiTreeCapture`] guard drop). Checked by every
/// `Drawable::prepaint`/`Interactivity::paint` call site before touching the
/// thread-local recorder, mirroring `capture_enabled()`'s (`flamegraph.rs`)
/// single-relaxed-atomic-load zero-overhead-when-idle discipline.
static UI_TREE_CAPTURE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// The most recently completed UI-tree capture, if any and if it has not
/// already been taken by [`take_completed_ui_tree_capture`].
static COMPLETED_UI_TREE_CAPTURE: parking_lot::Mutex<Option<UiTreeCapture>> = parking_lot::Mutex::new(None);

/// Arm a one-shot UI-tree capture: element tree, layout, style, and Scene
/// data for the very next `Window::draw()` call that actually draws (a
/// window with drawing skipped, e.g. `cx.mode.skip_drawing()`, does not
/// consume the request). Independent of Phase 1's always-on `Capture`
/// session and of Phase 4's `DeepCapture` -- see this module's doc comment
/// for why. Calling this while a capture is already armed or actively
/// recording is a no-op; the in-flight one is left to finish.
///
/// Not scoped to a specific window: in a multi-window app, whichever window
/// draws next (on GPUI's single foreground thread) consumes the request,
/// mirroring `request_deep_capture`'s (Phase 4) same not-window-scoped
/// behavior.
pub fn request_ui_tree_capture() {
    UI_TREE_CAPTURE_REQUESTED.store(true, Ordering::Release);
}

/// Whether a UI-tree capture has been requested but not yet started
/// recording.
pub fn ui_tree_capture_requested() -> bool {
    UI_TREE_CAPTURE_REQUESTED.load(Ordering::Acquire)
}

fn take_ui_tree_capture_request() -> bool {
    UI_TREE_CAPTURE_REQUESTED.swap(false, Ordering::AcqRel)
}

/// Take the most recently completed UI-tree capture, if any, clearing it so
/// a second call returns `None` until another capture finishes.
pub fn take_completed_ui_tree_capture() -> Option<UiTreeCapture> {
    COMPLETED_UI_TREE_CAPTURE.lock().take()
}

fn complete_ui_tree_capture(capture: UiTreeCapture) {
    *COMPLETED_UI_TREE_CAPTURE.lock() = Some(capture);
}

struct UiTreeRecorderState {
    window_id: u64,
    depth: u16,
    nodes: Vec<UiElementNode>,
    pending_styles: HashMap<u64, UiStyleSnapshot>,
}

impl UiTreeRecorderState {
    fn new(window_id: u64) -> Self {
        Self {
            window_id,
            depth: 0,
            nodes: Vec::new(),
            pending_styles: HashMap::new(),
        }
    }
}

thread_local! {
    static UI_TREE_RECORDER: RefCell<Option<UiTreeRecorderState>> = const { RefCell::new(None) };
}

/// RAII handle for an in-progress UI-tree capture recording, returned by
/// [`maybe_begin_capture`] and consumed by [`finish_capture`]. Its `Drop`
/// impl always clears `UI_TREE_CAPTURE_ACTIVE` and discards the thread-local
/// recorder, so a panic unwinding through `Window::draw_roots` (between
/// `maybe_begin_capture` and the normal `finish_capture` call) can't leave
/// the "actively recording" flag stuck true for the rest of the process --
/// `finish_capture` takes this by value and its own cleanup happens first,
/// making the `Drop` a harmless no-op on the success path.
pub(crate) struct ActiveUiTreeCapture {
    _private: (),
}

impl Drop for ActiveUiTreeCapture {
    fn drop(&mut self) {
        UI_TREE_CAPTURE_ACTIVE.store(false, Ordering::Release);
        UI_TREE_RECORDER.with(|recorder| {
            recorder.borrow_mut().take();
        });
    }
}

/// Consume a pending capture request, if any, and start recording on the
/// calling thread. Called from `Window::draw`, only on the path that is
/// actually about to call `draw_roots` (see [`request_ui_tree_capture`]'s
/// doc comment for why a skipped draw doesn't consume the request).
pub(crate) fn maybe_begin_capture(window_id: u64) -> Option<ActiveUiTreeCapture> {
    if !take_ui_tree_capture_request() {
        return None;
    }
    UI_TREE_RECORDER.with(|recorder| {
        *recorder.borrow_mut() = Some(UiTreeRecorderState::new(window_id));
    });
    UI_TREE_CAPTURE_ACTIVE.store(true, Ordering::Release);
    Some(ActiveUiTreeCapture { _private: () })
}

/// Finish recording, join style data onto tree nodes, deep-copy the finished
/// `Scene`, and publish the result. Called from `Window::draw` once
/// `next_frame.scene` has been sorted (`Frame::finish`), consuming the
/// [`ActiveUiTreeCapture`] guard [`maybe_begin_capture`] returned.
pub(crate) fn finish_capture(_guard: ActiveUiTreeCapture, scene: &Scene, scale_factor: f32) {
    let state = UI_TREE_RECORDER.with(|recorder| recorder.borrow_mut().take());
    UI_TREE_CAPTURE_ACTIVE.store(false, Ordering::Release);

    let Some(state) = state else {
        return;
    };

    let nodes = state
        .nodes
        .into_iter()
        .map(|mut node| {
            if node.global_id_hash != 0
                && let Some(style) = state.pending_styles.get(&node.global_id_hash)
            {
                node.style = Some(style.clone());
            }
            node
        })
        .collect();

    complete_ui_tree_capture(UiTreeCapture {
        window_id: state.window_id,
        nodes,
        scene: SceneSnapshot::capture(scene, scale_factor),
    });
}

/// RAII guard returned by [`record_element_prepaint`]. Dropping it restores
/// the recorder's traversal depth, mirroring `SpanGuard`'s (`flamegraph.rs`)
/// scoped push/pop -- this keeps nesting correct even if a descendant
/// element's `prepaint` panics, since `Drop` still runs during unwind.
pub(crate) struct UiTreeCaptureGuard {
    active: bool,
}

impl Drop for UiTreeCaptureGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        UI_TREE_RECORDER.with(|recorder| {
            if let Some(state) = recorder.borrow_mut().as_mut() {
                state.depth = state.depth.saturating_sub(1);
            }
        });
    }
}

/// Record one element's tree-structure entry during `Drawable::prepaint`
/// (`element.rs`), if a UI-tree capture is actively recording. Returns a
/// no-op guard immediately (no thread-local access at all) when no capture
/// is active, matching `enter_span`'s (Phase 1) zero-overhead-when-idle
/// discipline.
pub(crate) fn record_element_prepaint<E: Element>(
    global_id: Option<&GlobalElementId>,
    bounds: Bounds<Pixels>,
) -> UiTreeCaptureGuard {
    if !UI_TREE_CAPTURE_ACTIVE.load(Ordering::Relaxed) {
        return UiTreeCaptureGuard { active: false };
    }

    UI_TREE_RECORDER.with(|recorder| {
        if let Some(state) = recorder.borrow_mut().as_mut() {
            let depth = state.depth;
            let global_id_hash = global_id.map(crate::hash_global_element_id).unwrap_or(0);
            state.nodes.push(UiElementNode {
                type_name: type_name::<E>(),
                global_id_hash,
                depth,
                bounds: logical_bounds(bounds),
                style: None,
            });
            state.depth = state.depth.saturating_add(1);
        }
    });

    UiTreeCaptureGuard { active: true }
}

/// Record one `Interactivity`-backed element's resolved style during
/// `Interactivity::paint` (`elements/div.rs`), if a UI-tree capture is
/// actively recording and the element has a stable id. Joined onto the
/// matching [`UiElementNode`] by [`finish_capture`].
pub(crate) fn record_element_style(global_id: Option<&GlobalElementId>, style: &Style) {
    if !UI_TREE_CAPTURE_ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    let Some(global_id) = global_id else {
        return;
    };
    let hash = crate::hash_global_element_id(global_id);
    if hash == 0 {
        return;
    }

    let snapshot = UiStyleSnapshot::from_style(style);
    UI_TREE_RECORDER.with(|recorder| {
        if let Some(state) = recorder.borrow_mut().as_mut() {
            state.pending_styles.insert(hash, snapshot);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        self as gpui, Context, InteractiveElement, IntoElement, ParentElement, Render, Styled,
        TestAppContext, Window, div, rgb,
    };

    struct TestView;

    impl Render for TestView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div()
                .id("root")
                .size_full()
                .child(div().id("child-a").w(crate::px(40.0)).h(crate::px(20.0)).bg(rgb(0xff0000)))
                .child(div().id("child-b").w(crate::px(10.0)).h(crate::px(10.0)))
        }
    }

    #[gpui::test]
    fn triggered_capture_records_tree_layout_style_and_scene(cx: &mut TestAppContext) {
        // No capture requested yet: a normal draw should not record anything.
        let (_view, cx) = cx.add_window_view(|_, _| TestView);
        assert!(take_completed_ui_tree_capture().is_none());

        assert!(!ui_tree_capture_requested());
        request_ui_tree_capture();
        assert!(ui_tree_capture_requested());

        cx.update(|window, cx| {
            window.draw(cx).clear();
        });

        assert!(
            !ui_tree_capture_requested(),
            "the request should be consumed by the draw above"
        );

        let capture = take_completed_ui_tree_capture().expect("a capture should have completed");
        assert!(
            take_completed_ui_tree_capture().is_none(),
            "taking the completed capture should clear the mailbox"
        );

        assert!(!capture.nodes.is_empty(), "the tree should have recorded at least the test view's elements");

        // Every element with a stable id (`.id("root")`/`.id("child-a")`/
        // `.id("child-b")`) is `Interactivity`-backed, so all three should
        // have a resolved style captured -- look them up by that rather than
        // by reconstructing an exact `GlobalElementId` path, since this test
        // shouldn't need to hard-code how many wrapper elements GPUI's own
        // view/root plumbing puts around the returned `div()` tree.
        let styled_nodes: Vec<&UiElementNode> =
            capture.nodes.iter().filter(|node| node.style.is_some()).collect();
        assert!(
            styled_nodes.len() >= 3,
            "expected at least root/child-a/child-b to have resolved styles, got {styled_nodes:?}"
        );

        let expected_color: Hsla = rgb(0xff0000).into();
        let child_a = styled_nodes
            .iter()
            .find(|node| {
                node.style
                    .as_ref()
                    .and_then(|style| style.background.as_ref())
                    .and_then(Fill::color)
                    == Some(Background::from(expected_color))
            })
            .expect("child-a's red background should be captured on some node");

        let root = styled_nodes
            .iter()
            .min_by_key(|node| node.depth)
            .expect("at least one styled node should exist");
        assert!(root.bounds.width > 0.0 && root.bounds.height > 0.0, "root bounds should be resolved");
        assert!(
            child_a.depth > root.depth,
            "child-a (depth {}) should nest under the outermost styled node (depth {})",
            child_a.depth,
            root.depth
        );

        assert!(
            capture
                .scene
                .quads
                .iter()
                .any(|quad| quad.background == Background::from(expected_color)),
            "the scene snapshot should include child-a's background quad, got {:?}",
            capture.scene.quads
        );
    }
}
