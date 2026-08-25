//! Overscroll buffers for virtualized lists inside texture-retained layers
//! (#96).
//!
//! A list wrapped in a `.layer_with_policy(LayerPolicy { overdraw_margin, .. })`
//! div gets a persistent texture covering `viewport + 2 × margin`. Between
//! refills, scrolling costs one shifted surface draw — no re-record, no item
//! layout — and the layer re-renders only once accumulated scroll passes half
//! the margin, so the shift never outruns the texture.
//!
//! This module is the decision half of that protocol, shared by every
//! virtualized list: given the element's current scroll position it tells the
//! list whether to skip its per-item work entirely (the layer composites
//! shifted), lay out the full buffer range (a refill is re-recording the
//! layer), or fall back to the plain viewport range (no usable buffer).

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
