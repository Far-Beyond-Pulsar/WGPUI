//! Overscroll buffers for scrolled content inside texture-retained layers
//! (#96).
//!
//! A subtree wrapped in a `.layer_keyed(..)` +
//! `.layer_with_policy(LayerPolicy { overdraw_margin, .. })` div gets a
//! persistent texture covering `viewport + 2 × margin`. Between refills,
//! scrolling costs one shifted surface draw — no re-record, no item layout —
//! and the layer re-renders only once accumulated scroll passes half the
//! margin, so the shift never outruns the texture.
//!
//! This module is the decision half of that protocol: given an element's
//! current scroll position it tells the caller whether to skip its per-item
//! work entirely (the layer composites shifted), lay out the full buffer range
//! (a refill is re-recording the layer), or fall back to the plain viewport
//! range (no usable buffer).
//!
//! Callers are the virtualized lists (which skip per-item layout) and, since
//! this protocol left `virtual_list.rs`, plain scroll containers — a
//! `.overflow_scroll()` div under its own buffered keyed layer skips its
//! whole child prepaint on shift frames.
//!
//! **This only bounds the cost of a *shift* frame, not a *refill* frame, and
//! the difference matters a great deal for plain (non-virtualized) divs.**
//! `ScrollBufferFrame::Buffer`/`Viewport` say "lay out the buffer range," but
//! a virtualized list and a plain div satisfy that instruction very
//! differently: a virtualized list *synthesizes* only the rows inside the
//! range, so refill cost is bounded by the margin regardless of the
//! underlying dataset's size. A plain div's `children` is already a fully
//! materialized `Vec<AnyElement>` — every child that exists, exists as a real
//! element whether or not it's inside the buffer range — so `div.rs`'s
//! non-`Skip` branch lays out *all* of them, every refill (first mount, every
//! resize, every scroll past the margin). Measured on a 10,000-row plain div:
//! ~1.3s per refill (docs/scroll-free-by-default.md §0.-1). A buffered plain
//! div is a good fit for small-to-moderate child counts where "lay out
//! everything" was always going to be cheap; for large real lists, use
//! `uniform_list`/`virtual_list`/`h_list` instead. `div.rs` warns once, in
//! debug builds, when a buffered scroll container's child count crosses a
//! threshold that makes this the wrong tool.

use crate::{LayerKey, Pixels, Point, Size, point, px};

/// What a list's `prepaint` should do this frame.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ScrollBufferFrame {
    /// The enclosing layer composites from its texture, shifted by the
    /// content offset already recorded on the layer: skip per-item layout
    /// entirely. Hitboxes and the scroll listener stay alive through the
    /// layer's recorded paint-range replay.
    Skip,
    /// Lay out items across the full buffer range (`viewport + 2 × margin`)
    /// and paint them under the buffer's content mask. The layer's texture
    /// is being (re)rendered this frame.
    Buffer { margin: Size<Pixels> },
    /// No usable buffer this frame; lay out the viewport range as before.
    Viewport,
}

/// Decide the frame for a list at scroll position `scroll` (the smoothed
/// visual scroll the items will be painted at).
///
/// Must be called from `prepaint`, inside the layer's hitbox scope, after the
/// scroll offset is final: the prediction it makes about the layer's paint
/// reads the same frame state the paint-time decision will.
pub(crate) fn prepare_scroll_buffer(
    window: &mut crate::Window,
    scroll: Point<Pixels>,
) -> ScrollBufferFrame {
        let Some(buffer) = window.prepaint_layer_buffer() else {
        return ScrollBufferFrame::Viewport;
    };
    let key: LayerKey = buffer.key;

    // A refill re-renders the texture re-centred on the current scroll: the
    // content offset resets and the anchor moves with it.
    if buffer.refilling {
        window.set_layer_buffer_anchor(key, scroll);
        window.set_layer_content_offset(key, Point::default());
        return ScrollBufferFrame::Buffer {
            margin: buffer.margin,
        };
    }

    // The texture does not cover the buffer yet (the first record ran before
    // the layer existed at prepaint time). Ask for a refill and keep painting
    // the viewport range until it lands.
    if !buffer.buffer_ready {
        window.request_layer_buffer_refill(key);
        return ScrollBufferFrame::Viewport;
    }

    // How far the content scrolled since the buffer was rendered. The buffer
    // extends `margin` beyond the viewport on each side, so a shift of up to
    // ±margin still samples inside the texture.
    let delta_x = scroll.x - buffer.anchor.x;
    let delta_y = scroll.y - buffer.anchor.y;
    let half_x = buffer.margin.width / 2.0;
    let half_y = buffer.margin.height / 2.0;
    let exceeded = (buffer.margin.width > px(0.) && delta_x.abs() > half_x)
        || (buffer.margin.height > px(0.) && delta_y.abs() > half_y);

    // Clamp so the composite never samples outside the texture even while a
    // requested refill is still one frame in flight.
    let clamped = point(
        delta_x.clamp(-buffer.margin.width, buffer.margin.width),
        delta_y.clamp(-buffer.margin.height, buffer.margin.height),
    );
    window.set_layer_content_offset(key, clamped);
    if exceeded {
        // Half a margin of slack: the refill lands before the shift can
        // reach the texture edge, so no frame ever shows a blank band.
        window.request_layer_buffer_refill(key);
    }

    if buffer.will_composite {
        return ScrollBufferFrame::Skip;
    }

    // The prediction says this frame records after all (a resize, an accessed
    // dependency, a pointer-condition re-render). Re-anchor so the fresh
    // texture matches what the list is about to paint.
    window.set_layer_buffer_anchor(key, scroll);
    window.set_layer_content_offset(key, Point::default());
    ScrollBufferFrame::Buffer {
        margin: buffer.margin,
    }
}

/// The item range covering content-space `from..to`, via the same
/// partition-point lookup the list uses for its viewport range.
pub(crate) fn inflate_for_buffer(
    bounds: crate::Bounds<Pixels>,
    margin: Size<Pixels>,
) -> crate::Bounds<Pixels> {
    crate::layer::inflate_bounds(bounds, margin)
}

// ---------------------------------------------------------------------
// Layout containment for plain (non-virtualized) scroll containers (#96,
// docs/scroll-free-by-default.md §0.-2).
// ---------------------------------------------------------------------

/// What `Div::request_layout` should do with one child of a buffered scroll
/// container: give it a real layout, or stand it in with a placeholder sized
/// from [`crate::Element::estimated_size`] — the same idea as CSS
/// `content-visibility: auto` + `contain-intrinsic-size`. A contained child's
/// own `request_layout`/`prepaint`/`paint` never run: no instance
/// reconciliation, no style resolution, no recursion into its subtree at all.
/// Only its size (already known some cheap way) contributes to the parent's
/// content extent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ChildContainment {
    /// Outside the visible+margin window; use this size and skip everything
    /// else.
    Contained(Size<Pixels>),
    /// Inside the window, or its position couldn't be tracked cheaply enough
    /// to know — lay it out for real, exactly as an unbuffered div would.
    Real,
}

/// Decides containment for a vertical scroll container's children, one at a
/// time, in order — the stack-only, single-pass replacement for what used to
/// build a whole `Vec<ChildContainment>` up front (one full pass to build it,
/// a second to consume it, plus a heap allocation for a list as long as the
/// child count — real overhead at 10,000 children, paid every single frame
/// regardless of whether anything changed). A caller now folds one
/// `.decide(estimated_size)` call into whatever loop it already runs over
/// its children; nothing here allocates.
///
/// **Position is tracked by accumulation, not measurement** — this runs at
/// `request_layout` time, before Taffy has computed anything, so a child's
/// true position depends on the declared heights of every child before it,
/// not their real ones (which aren't known yet regardless of whether this
/// child ends up `Contained` or `Real`). The first child with no
/// `estimated_size` breaks that chain: its real height isn't known until
/// Taffy actually computes the tree, so nothing after it can be positioned
/// either, and the whole remaining suffix falls back to `Real` — exactly
/// what every child got before this existed, never worse. This is a
/// disclosed limitation of accumulating from style alone, not a correctness
/// gap: reusing a child's *last frame's* real bounds as a fallback estimate
/// (the CSS analogy's actual behavior for previously-rendered content) would
/// recover most of this and is a natural follow-up, not implemented here.
pub(crate) struct ContainmentWindow {
    window_top: Pixels,
    window_bottom: Pixels,
    running_y: Pixels,
    broken: bool,
}

impl ContainmentWindow {
    /// `scroll_offset_y` follows this crate's convention: zero or negative,
    /// more negative the further the content has scrolled down.
    /// `viewport_height` and `margin_height` both come from the enclosing
    /// layer/scroll state exactly as the shift-frame protocol above already
    /// uses them.
    pub(crate) fn new(scroll_offset_y: Pixels, viewport_height: Pixels, margin_height: Pixels) -> Self {
        Self {
            window_top: -scroll_offset_y - margin_height,
            window_bottom: -scroll_offset_y + viewport_height + margin_height,
            running_y: px(0.),
            broken: false,
        }
    }

    /// Decide the next child in order. Must be called for every child, in
    /// the same order every frame — the accumulation depends on it.
    pub(crate) fn decide(&mut self, estimated: Option<Size<Pixels>>) -> ChildContainment {
        if self.broken {
            return ChildContainment::Real;
        }
        let Some(size) = estimated else {
            self.broken = true;
            return ChildContainment::Real;
        };
        let top = self.running_y;
        let bottom = self.running_y + size.height;
        self.running_y = bottom;
        if bottom < self.window_top || top > self.window_bottom {
            ChildContainment::Contained(size)
        } else {
            ChildContainment::Real
        }
    }
}
