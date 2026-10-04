//! In-place re-rendering of isolation boundaries ([`AnyView::isolated`]).
//!
//! # The problem
//!
//! `cx.notify()` on a view marks every view above it dirty, and every cached
//! ancestor's dependency set is cumulative over its subtree, so it fails its
//! cache test too. A leaf that changes ten times a second therefore rebuilds the
//! whole chain of panels above it ten times a second, which is exactly what
//! the retained-layer work was meant to end (`docs/retained-layers.md`, "No
//! unconditional upward propagation") but nothing could yet request.
//!
//! # What an isolation boundary does
//!
//! A cached view that asked to be isolated is its own retained layer already.
//! Its ancestors do not need to *rebuild* to show new content, only to
//! *composite* its layer, and compositing reads the layer's items at the moment
//! it happens. So:
//!
//! 1. The boundary's dependencies are not folded into its ancestors' sets, and
//!    the dirty walk ends at it ([`Window::mark_view_dirty`]).
//! 2. At the start of the draw, [`Window::rerecord_isolated_views`] re-renders
//!    each invalidated boundary on its own: `render`, layout at its remembered
//!    bounds, prepaint, paint into its layer. The frame arrays the walk is about
//!    to build are not touched; the work happens in a scratch frame whose only
//!    product is the layer's new items.
//! 3. The walk then replays the ancestors as usual, and their composite picks
//!    the new items up.
//!
//! # What is deliberately not refreshed
//!
//! Hitboxes, listeners, tooltips, focus and dispatch nodes: the scratch frame is
//! thrown away, so the boundary's interactive surface stays as the last walk
//! recorded it. That is why a boundary has to be display-only, and why a height
//! change (which the ancestors must lay out for) falls back to the ordinary
//! path instead of being absorbed here.

use super::*;
use crate::AnyWeakView;

/// What the window remembers about an isolation boundary between visits by the
/// element walk. Written by the walk, read by the in-place re-render.
pub(crate) struct IsolatedViewRecord {
    view: AnyWeakView,
    global_id: GlobalElementId,
    pub(crate) layer_key: LayerKey,
    bounds: Bounds<Pixels>,
    content_mask: ContentMask<Pixels>,
    text_style_stack: Vec<TextStyleRefinement>,
    /// Entities the view reads. Kept here, out of its ancestors' sets, and still
    /// reported to the window so a notify on any of them finds something to
    /// invalidate.
    pub(crate) accessed: FxHashSet<EntityId>,
    auto_height: bool,
}

impl Window {
    /// Whether `view` was already re-rendered in place during this draw.
    pub(crate) fn isolated_view_is_fresh(&self, view: EntityId) -> bool {
        self.isolated_fresh.contains(&view)
    }

    /// The next invalidation of `view` must reach its ancestors (its size
    /// changed, so they have to lay out again).
    pub(crate) fn request_isolated_relayout(&mut self, view: EntityId) {
        self.isolated_relayout.insert(view);
    }

    /// Record that the element walk visited an isolation boundary.
    ///
    /// `rebuilt` is false for a cache hit, which can keep the text style stack it
    /// already has: a change there would have been a miss.
    pub(crate) fn note_isolated_view(
        &mut self,
        view: &AnyView,
        global_id: &GlobalElementId,
        bounds: Bounds<Pixels>,
        content_mask: ContentMask<Pixels>,
        accessed: FxHashSet<EntityId>,
        auto_height: bool,
        rebuilt: bool,
    ) {
        let entity = view.entity_id();
        let layer_key = LayerKey::from_global_element_id(global_id);
        match self.isolated_views.get_mut(&entity) {
            Some(record) => {
                record.bounds = bounds;
                record.content_mask = content_mask;
                record.accessed = accessed;
                record.auto_height = auto_height;
                record.layer_key = layer_key;
                if rebuilt || record.global_id != *global_id {
                    record.global_id = global_id.clone();
                    record.text_style_stack = self.text_style_stack.clone();
                }
            }
            None => {
                self.isolated_views.insert(
                    entity,
                    IsolatedViewRecord {
                        view: view.downgrade(),
                        global_id: global_id.clone(),
                        layer_key,
                        bounds,
                        content_mask,
                        text_style_stack: self.text_style_stack.clone(),
                        accessed,
                        auto_height,
                    },
                );
            }
        }
    }

    /// Re-render, in place, every isolation boundary that was invalidated since
    /// the last draw. Runs before the element walk.
    pub(crate) fn rerecord_isolated_views(&mut self, cx: &mut App) {
        if self.isolated_views.is_empty() {
            return;
        }
        // A boundary whose layer is gone (evicted, or its view left the tree)
        // has nothing to patch.
        let layers = &self.layers;
        self.isolated_views
            .retain(|_, record| layers.contains_key(&record.layer_key));

        let due: Vec<EntityId> = self
            .isolated_views
            .iter()
            .filter(|(id, record)| {
                self.dirty_views.contains(id)
                    || self.accessed_entity_invalidated(&record.accessed)
                    // The renderer asked for this layer to be recorded again (slab
                    // eviction or overflow): nothing else would, while the walk
                    // replays its ancestors.
                    || self
                        .layers
                        .get(&record.layer_key)
                        .is_some_and(|layer| !layer.needs.is_empty())
            })
            .map(|(id, _)| *id)
            .collect();
        if due.is_empty() {
            return;
        }

        let can_patch = crate::layer::layers_enabled()
            && self.view_cache_available()
            && !self.is_inspector_picking(cx);

        for id in due {
            if can_patch && self.rerecord_isolated_view(id, cx) {
                crate::render_stats::count("isolated view: re-rendered in place");
            } else {
                crate::render_stats::count("isolated view: fell back to ancestors");
                self.invalidate_through_ancestors(id);
            }
        }
    }

    /// Treat `view` as an ordinary view for this draw: dirty everything above it
    /// so the walk rebuilds the whole chain.
    fn invalidate_through_ancestors(&mut self, view: EntityId) {
        self.dirty_views.remove(&view);
        self.isolated_relayout.insert(view);
        self.mark_view_dirty(view);
        self.isolated_relayout.remove(&view);
        self.dirty_views.insert(view);
    }

    /// Re-render one boundary into its layer. Returns false, having changed
    /// nothing visible, when it cannot be done safely.
    fn rerecord_isolated_view(&mut self, id: EntityId, cx: &mut App) -> bool {
        let Some(record) = self.isolated_views.get(&id) else {
            return true;
        };
        let Some(view) = record.view.upgrade() else {
            self.isolated_views.remove(&id);
            return true;
        };
        let Some(layer) = self.layers.get(&record.layer_key) else {
            return false;
        };
        // Only a layer that holds primitive content and sits where it was last
        // recorded can be patched: a texture-retained one has baked pixels, and
        // one that moved is the walk's business.
        if !layer.has_content()
            || layer.texture_retained
            || layer.cache_key.bounds != record.bounds
        {
            return false;
        }

        let layer_key = record.layer_key;
        let cache_key = layer.cache_key.clone();
        let bounds = record.bounds;
        let auto_height = record.auto_height;
        let global_id = record.global_id.clone();
        let mask = record.content_mask.clone();
        let text_style_stack = record.text_style_stack.clone();

        // Everything this does to the frame lands in a scratch frame.
        let mut scratch = self.isolated_scratch.take().unwrap_or_else(|| {
            Box::new(Frame::new(DispatchTree::new(
                cx.keymap.clone(),
                cx.actions.clone(),
            )))
        });
        mem::swap(&mut self.next_frame, &mut *scratch);
        let saved_ids = mem::replace(
            &mut self.element_id_stack,
            global_id.0.iter().cloned().collect(),
        );
        let saved_text = mem::replace(&mut self.text_style_stack, text_style_stack);
        let saved_mask = mem::replace(&mut self.content_mask_stack, vec![mask]);
        // Opacity multiplies into every colour at paint time, so it has to be what
        // the walk would have had.
        let saved_opacity = mem::replace(&mut self.element_opacity, cache_key.opacity);
        let phase = self.invalidator.draw_phase();

        let _arena_scope = ElementArenaScope::enter(cx.element_arena());
        self.invalidator.set_phase(DrawPhase::Prepaint);
        let ((element, measured), accessed) = cx.detect_accessed_entities_with(false, |cx| {
            self.with_rendered_view(id, |window| {
                let mut element = view.render_element(window, cx);
                let measured = if auto_height {
                    // Content height at this width, as the walk measures it.
                    let available = size(
                        AvailableSpace::Definite(bounds.size.width),
                        AvailableSpace::MaxContent,
                    );
                    Some(element.layout_as_root(available, window, cx).height)
                } else {
                    element.layout_as_root(bounds.size.into(), window, cx);
                    None
                };
                element.prepaint_at(bounds.origin, window, cx);
                (element, measured)
            })
        });

        // A different height is for the ancestors to lay out around.
        let resized = measured.is_some_and(|height| (height - bounds.size.height).abs() > px(0.5));
        let mut element = element;
        if !resized {
            self.invalidator.set_phase(DrawPhase::Paint);
            self.with_rendered_view(id, |window| {
                window.record_layer(layer_key, cache_key, LayerPolicy::compat(), |window| {
                    element.paint(window, cx)
                });
            });
        }

        self.element_id_stack = saved_ids;
        self.text_style_stack = saved_text;
        self.content_mask_stack = saved_mask;
        self.element_opacity = saved_opacity;
        mem::swap(&mut self.next_frame, &mut *scratch);

        // Element state the subtree used moved from the previous frame into the
        // scratch one; hand it to the real frame, or it would be swept.
        self.next_frame
            .element_states
            .extend(scratch.element_states.drain());
        self.next_frame
            .accessed_element_states
            .extend(scratch.accessed_element_states.drain(..));
        scratch.clear();
        self.isolated_scratch = Some(scratch);

        if resized {
            self.invalidator.set_phase(phase);
            return false;
        }

        // The view's stored state must describe what it reads now.
        self.invalidator.set_phase(DrawPhase::Prepaint);
        crate::view::refresh_view_dependencies(&global_id, accessed.clone(), self);
        self.invalidator.set_phase(phase);

        if let Some(record) = self.isolated_views.get_mut(&id) {
            record.accessed = accessed;
        }
        self.isolated_fresh.insert(id);
        self.dirty_views.remove(&id);
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        AnyView, Context, Entity, IntoElement, Render, StyleRefinement, TestAppContext, Window,
        blue, canvas, div, fill, prelude::*, px, size,
    };
    use std::cell::Cell;
    use std::rc::Rc;

    /// Counters the views bump so the tests can see who rendered and who painted.
    #[derive(Default)]
    struct Counts {
        mid_renders: Cell<usize>,
        leaf_renders: Cell<usize>,
        leaf_paints: Cell<usize>,
        /// The value the leaf last painted.
        painted: Cell<u32>,
    }

    struct Leaf {
        value: u32,
        counts: Rc<Counts>,
    }

    impl Render for Leaf {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.counts.leaf_renders.set(self.counts.leaf_renders.get() + 1);
            let value = self.value;
            let counts = self.counts.clone();
            div().w_full().h(px(20.)).bg(crate::red().opacity(0.3)).child(
                canvas(
                    |_, _, _| (),
                    move |bounds, _, window, _| {
                        counts.leaf_paints.set(counts.leaf_paints.get() + 1);
                        counts.painted.set(value);
                        window.paint_quad(fill(bounds, blue()));
                    },
                )
                .size_full(),
            )
        }
    }

    struct Mid {
        leaf: Entity<Leaf>,
        isolated: bool,
        auto_height: bool,
        counts: Rc<Counts>,
    }

    impl Render for Mid {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.counts.mid_renders.set(self.counts.mid_renders.get() + 1);
            let leaf = AnyView::from(self.leaf.clone());
            let leaf = if self.auto_height {
                leaf.cached_auto_height(StyleRefinement::default().w(px(40.)))
            } else {
                leaf.cached(StyleRefinement::default().w(px(40.)).h(px(20.)))
            };
            div()
                .size_full()
                .child(canvas(|_, _, _| (), |bounds, _, window, _| window.paint_quad(fill(bounds, crate::red().opacity(0.5)))).absolute().top_0().left_0().w(px(100.)).h(px(100.)))
                .child(if self.isolated { leaf.isolated() } else { leaf })
        }
    }

    struct Root {
        mid: Entity<Mid>,
    }

    impl Render for Root {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child(
                AnyView::from(self.mid.clone())
                    .cached(StyleRefinement::default().w(px(100.)).h(px(100.))),
            )
        }
    }

    fn tree(
        cx: &mut TestAppContext,
        isolated: bool,
    ) -> (crate::AnyWindowHandle, Entity<Leaf>, Rc<Counts>) {
        tree_with(cx, isolated, false)
    }

    fn tree_with(
        cx: &mut TestAppContext,
        isolated: bool,
        auto_height: bool,
    ) -> (crate::AnyWindowHandle, Entity<Leaf>, Rc<Counts>) {
        let counts = Rc::new(Counts::default());
        let leaf = cx.update(|cx| {
            cx.new(|_| Leaf {
                value: 0,
                counts: counts.clone(),
            })
        });
        let mid = cx.update(|cx| {
            cx.new(|_| Mid {
                leaf: leaf.clone(),
                isolated,
                auto_height,
                counts: counts.clone(),
            })
        });
        let window = cx.open_window(size(px(400.), px(300.)), move |_, _| Root { mid });
        cx.run_until_parked();
        let window: crate::AnyWindowHandle = window.into();
        // Quiet frames so every cached view has settled into its reuse path.
        for _ in 0..6 {
            window.update(cx, |_, window, _| window.refresh_buffers()).unwrap();
            cx.run_until_parked();
        }
        (window, leaf, counts)
    }

    fn set_value(cx: &mut TestAppContext, leaf: &Entity<Leaf>, value: u32) {
        leaf.update(cx, |leaf, cx| {
            leaf.value = value;
            cx.notify();
        });
        cx.run_until_parked();
    }

    /// The control: without isolation a notify rebuilds the whole chain above.
    #[gpui::test]
    fn an_ordinary_cached_view_rebuilds_its_parent(cx: &mut TestAppContext) {
        let (_window, leaf, counts) = tree(cx, false);
        let mid_before = counts.mid_renders.get();
        set_value(cx, &leaf, 1);
        assert!(counts.mid_renders.get() > mid_before);
        assert_eq!(counts.painted.get(), 1);
    }

    /// The point: an isolated view refreshes by itself.
    #[gpui::test]
    fn an_isolated_view_refreshes_without_rebuilding_its_ancestors(cx: &mut TestAppContext) {
        if !crate::layer::layers_enabled() {
            return;
        }
        let (_window, leaf, counts) = tree(cx, true);
        let (mid, renders, paints) = (
            counts.mid_renders.get(),
            counts.leaf_renders.get(),
            counts.leaf_paints.get(),
        );

        for value in 1..=5 {
            set_value(cx, &leaf, value);
            assert_eq!(
                counts.mid_renders.get(),
                mid,
                "value {value}: an ancestor rendered"
            );
            assert_eq!(
                counts.painted.get(),
                value,
                "value {value}: the new content never reached the layer"
            );
        }
        assert_eq!(counts.leaf_renders.get(), renders + 5, "one render per change");
        assert_eq!(counts.leaf_paints.get(), paints + 5, "one paint per change");
    }

    /// A frame with nothing invalidated afterwards must not redo the work, and
    /// the content must stay what was last drawn.
    #[gpui::test]
    fn a_quiet_frame_after_an_isolated_refresh_replays(cx: &mut TestAppContext) {
        if !crate::layer::layers_enabled() {
            return;
        }
        let (window, leaf, counts) = tree(cx, true);
        set_value(cx, &leaf, 7);
        let (renders, paints, mid) = (
            counts.leaf_renders.get(),
            counts.leaf_paints.get(),
            counts.mid_renders.get(),
        );
        for _ in 0..3 {
            window.update(cx, |_, window, _| window.refresh_buffers()).unwrap();
            cx.run_until_parked();
        }
        assert_eq!(counts.leaf_renders.get(), renders);
        assert_eq!(counts.leaf_paints.get(), paints);
        assert_eq!(counts.mid_renders.get(), mid);
        assert_eq!(counts.painted.get(), 7);
    }

    /// An ancestor that does rebuild (here, notified directly) must still show
    /// the isolated view's latest content, not the content from when it last
    /// walked it.
    #[gpui::test]
    fn an_ancestor_rebuild_keeps_the_isolated_content(cx: &mut TestAppContext) {
        if !crate::layer::layers_enabled() {
            return;
        }
        let (window, leaf, counts) = tree(cx, true);
        set_value(cx, &leaf, 3);
        window.update(cx, |_, window, _| window.refresh()).unwrap();
        cx.run_until_parked();
        assert_eq!(counts.painted.get(), 3);
        set_value(cx, &leaf, 4);
        assert_eq!(counts.painted.get(), 4);
    }

    /// The overlay case: the view's height comes from its content
    /// (`cached_auto_height`), measured on the first inline render and
    /// remembered after.
    #[gpui::test]
    fn an_auto_height_isolated_view_refreshes_in_place(cx: &mut TestAppContext) {
        if !crate::layer::layers_enabled() {
            return;
        }
        let (_window, leaf, counts) = tree_with(cx, true, true);
        // The first change finds the view still rendered inline (it had no height to
        // cache against yet) and goes the ordinary way; from then on it is a boundary.
        set_value(cx, &leaf, 100);
        let mid = counts.mid_renders.get();
        for value in 1..=3 {
            set_value(cx, &leaf, value);
            assert_eq!(counts.painted.get(), value);
        }
        assert_eq!(counts.mid_renders.get(), mid, "an ancestor rendered");
    }

}
