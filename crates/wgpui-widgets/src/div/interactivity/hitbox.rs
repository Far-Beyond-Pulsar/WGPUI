//! Hitbox/dispatch-node registration for interactive divs.

use wgpui_core::geometry::Rect;
use wgpui_core::window::{DispatchNodeId, FocusHandle, Hitbox, HitboxId, Window};

/// Register the geometry and dispatch relationship for one interactive node.
/// Re-registering the same id updates its bounds without changing its paint
/// order, which keeps hit testing stable while layout changes.
pub fn register(
    window: &mut Window,
    id: HitboxId,
    bounds: Rect,
    z_index: i32,
    node: DispatchNodeId,
    focus: Option<FocusHandle>,
) {
    let hitbox = Hitbox {
        id,
        bounds,
        z_index,
        order: 0,
        hit_testable: true,
    };
    if let Some(focus) = focus {
        window.register_focus_hitbox(hitbox, node, focus);
    } else {
        window.register_hitbox(hitbox, node);
    }
}
