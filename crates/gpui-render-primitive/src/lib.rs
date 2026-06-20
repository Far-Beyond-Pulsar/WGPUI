//! Zero-overhead, extensible render-primitive plugin API for WGPUI.
//!
//! # Why this crate exists
//!
//! WGPUI's built-in primitives (quads, shadows, sprites, …) are compiled into the
//! core. New, diverse primitive types — e.g. a WGPU canvas — can instead live in
//! their **own crate** that depends only on *this* crate, so editing one primitive
//! does not recompile the others or the core.
//!
//! # Zero runtime overhead
//!
//! The hard part (per the maintainer) is "zero overhead at runtime while letting
//! primitive crates do effectively anything." This API solves it by dispatching
//! **once per batch, not per instance**:
//!
//! * A primitive crate implements [`RenderPrimitive`] for its type and registers a
//!   single instance in a [`PrimitiveRegistry`].
//! * Each frame, the painter appends raw, `Pod` instance bytes into a
//!   [`PrimitiveBatches`] keyed by [`PrimitiveTypeId`]. Appending is a `memcpy`;
//!   there is no per-instance virtual call and no boxing.
//! * At draw time the renderer makes exactly one dynamic call per *type* present
//!   ([`PrimitiveRegistry::draw_batch`]), which issues a single batched GPU draw
//!   for every instance of that type. Dispatch cost is therefore O(number of
//!   primitive *types*), which is negligible.
//!
//! GPU work (`build`/`draw`) is only ever invoked by the core renderer with a live
//! device; the registry and batch bookkeeping are GPU-free and unit-tested here.

use std::any::Any;
use std::collections::HashMap;

/// Re-exported so primitive crates and the core share one `wgpu` (and thus one set
/// of GPU types). Plugins must use `gpui_render_primitive::wgpu`, never their own.
pub use wgpu;

/// Screen-space bounds in scaled (device) pixels, used for damage tracking and
/// draw ordering. Kept minimal so this crate stays decoupled from the core's
/// geometry types.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PrimBounds {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl PrimBounds {
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.width <= 0.0 || self.height <= 0.0
    }

    /// Smallest bounds containing both `self` and `other`. Empty bounds act as the
    /// identity so a fresh batch's union starts from its first instance.
    pub fn union(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        let min_x = self.x.min(other.x);
        let min_y = self.y.min(other.y);
        let max_x = (self.x + self.width).max(other.x + other.width);
        let max_y = (self.y + self.height).max(other.y + other.height);
        Self {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        }
    }
}

/// Stable identity for a primitive *type*. Use a globally-unique string, e.g. the
/// fully-qualified type path (`"my_crate::Stripes"`).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PrimitiveTypeId(pub &'static str);

/// GPU context handed to [`RenderPrimitive::build`] to create the primitive's
/// pipeline and any persistent resources, once.
pub struct PipelineContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    /// Color format of the render target the primitive will draw into.
    pub surface_format: wgpu::TextureFormat,
    /// Bind-group layout for the shared per-frame globals (viewport size, etc.),
    /// so primitives can position themselves consistently with built-ins.
    pub globals_layout: &'a wgpu::BindGroupLayout,
}

/// GPU context handed to [`RenderPrimitive::draw`] to render one batch.
pub struct BatchDraw<'a, 'pass> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub pass: &'a mut wgpu::RenderPass<'pass>,
    /// The shared per-frame globals bind group (bind at group 0).
    pub globals: &'a wgpu::BindGroup,
    /// Raw, tightly-packed instance bytes for every instance of this type queued
    /// this frame, in submission order. The plugin defined the layout, so it
    /// knows how to reinterpret these (e.g. via `bytemuck::cast_slice`).
    pub instances: &'a [u8],
    /// Number of instances packed into `instances`.
    pub instance_count: u32,
}

/// A render-primitive type plugin. Implement this in your own crate for a custom
/// primitive; one instance is [`register`](PrimitiveRegistry::register)ed with the
/// core. All methods are batched: the core never calls per individual instance.
pub trait RenderPrimitive: Any + Send + Sync {
    /// Globally-unique id for this primitive type.
    fn type_key(&self) -> PrimitiveTypeId;

    /// Build the GPU pipeline and any persistent resources, once. The returned
    /// value is opaque to the core and is handed back to [`draw`](Self::draw).
    fn build(&self, ctx: &PipelineContext<'_>) -> Box<dyn Any + Send + Sync>;

    /// Draw every queued instance of this type in a single batched pass, using the
    /// pipeline state previously returned by [`build`](Self::build).
    fn draw(&self, state: &mut (dyn Any + Send + Sync), batch: BatchDraw<'_, '_>);
}

/// Registry of primitive-type plugins plus their lazily-built pipeline state.
/// Owned by the core renderer.
#[derive(Default)]
pub struct PrimitiveRegistry {
    kinds: HashMap<PrimitiveTypeId, Box<dyn RenderPrimitive>>,
    pipelines: HashMap<PrimitiveTypeId, Box<dyn Any + Send + Sync>>,
}

impl PrimitiveRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a primitive-type plugin. Re-registering the same type replaces it
    /// and drops any built pipeline so it is rebuilt on next use.
    pub fn register(&mut self, kind: Box<dyn RenderPrimitive>) {
        let id = kind.type_key();
        self.kinds.insert(id, kind);
        self.pipelines.remove(&id);
    }

    pub fn contains(&self, id: PrimitiveTypeId) -> bool {
        self.kinds.contains_key(&id)
    }

    pub fn registered_type_count(&self) -> usize {
        self.kinds.len()
    }

    /// Draw one batch: ensure the type's pipeline is built, then issue its single
    /// batched draw. This is the *only* dynamic dispatch per frame per type. No-op
    /// if the type is not registered (the core logs/skips unknown types).
    pub fn draw_batch(
        &mut self,
        id: PrimitiveTypeId,
        instances: &[u8],
        instance_count: u32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        globals_layout: &wgpu::BindGroupLayout,
        globals: &wgpu::BindGroup,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        let Some(kind) = self.kinds.get(&id) else {
            return false;
        };
        let state = self.pipelines.entry(id).or_insert_with(|| {
            kind.build(&PipelineContext {
                device,
                queue,
                surface_format,
                globals_layout,
            })
        });
        kind.draw(
            state.as_mut(),
            BatchDraw {
                device,
                queue,
                pass,
                globals,
                instances,
                instance_count,
            },
        );
        true
    }
}

/// One type's accumulated instances for a frame.
#[derive(Default)]
pub struct PrimitiveBatch {
    /// Tightly-packed `Pod` instance bytes in submission order.
    pub bytes: Vec<u8>,
    pub count: u32,
    /// Union of all instance bounds, for damage and draw ordering.
    pub bounds: PrimBounds,
}

/// Per-frame collection of custom-primitive instances, grouped by type so each
/// type can be drawn in a single batch. Lives on the scene; cleared each frame.
#[derive(Default)]
pub struct PrimitiveBatches {
    batches: HashMap<PrimitiveTypeId, PrimitiveBatch>,
    /// First-seen order of types, so draws are deterministic across frames.
    order: Vec<PrimitiveTypeId>,
}

impl PrimitiveBatches {
    /// Append one instance (its raw `Pod` bytes) to its type's batch.
    pub fn push(&mut self, id: PrimitiveTypeId, instance_bytes: &[u8], bounds: PrimBounds) {
        let batch = self.batches.entry(id).or_insert_with(|| {
            self.order.push(id);
            PrimitiveBatch::default()
        });
        batch.bytes.extend_from_slice(instance_bytes);
        batch.count += 1;
        batch.bounds = batch.bounds.union(bounds);
    }

    pub fn clear(&mut self) {
        self.batches.clear();
        self.order.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    pub fn type_count(&self) -> usize {
        self.order.len()
    }

    /// Iterate batches in first-seen (deterministic) order.
    pub fn iter(&self) -> impl Iterator<Item = (PrimitiveTypeId, &PrimitiveBatch)> {
        self.order
            .iter()
            .filter_map(move |id| self.batches.get(id).map(|batch| (*id, batch)))
    }

    /// Union of all batches' bounds, for whole-frame damage when custom primitives
    /// are present.
    pub fn total_bounds(&self) -> PrimBounds {
        self.batches
            .values()
            .fold(PrimBounds::default(), |acc, batch| acc.union(batch.bounds))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: PrimitiveTypeId = PrimitiveTypeId("test::A");
    const B: PrimitiveTypeId = PrimitiveTypeId("test::B");

    #[test]
    fn bounds_union_treats_empty_as_identity() {
        let empty = PrimBounds::default();
        let r = PrimBounds::new(10.0, 20.0, 30.0, 40.0);
        assert_eq!(empty.union(r), r);
        assert_eq!(r.union(empty), r);
        let joined =
            PrimBounds::new(0.0, 0.0, 10.0, 10.0).union(PrimBounds::new(20.0, 5.0, 10.0, 10.0));
        assert_eq!(joined, PrimBounds::new(0.0, 0.0, 30.0, 15.0));
    }

    #[test]
    fn batches_group_by_type_preserve_order_and_pack_bytes() {
        let mut batches = PrimitiveBatches::default();
        batches.push(A, &[1, 2, 3, 4], PrimBounds::new(0.0, 0.0, 4.0, 4.0));
        batches.push(B, &[9, 9], PrimBounds::new(10.0, 0.0, 2.0, 2.0));
        batches.push(A, &[5, 6, 7, 8], PrimBounds::new(0.0, 4.0, 4.0, 4.0));

        assert_eq!(batches.type_count(), 2);
        let collected: Vec<_> = batches
            .iter()
            .map(|(id, batch)| (id, batch.bytes.clone(), batch.count))
            .collect();
        // First-seen order: A then B.
        assert_eq!(collected[0].0, A);
        assert_eq!(collected[0].1, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(collected[0].2, 2);
        assert_eq!(collected[1].0, B);
        assert_eq!(collected[1].1, vec![9, 9]);
        assert_eq!(collected[1].2, 1);

        // A's bounds are the union of its two instances.
        let a_bounds = batches.iter().find(|(id, _)| *id == A).unwrap().1.bounds;
        assert_eq!(a_bounds, PrimBounds::new(0.0, 0.0, 4.0, 8.0));

        batches.clear();
        assert!(batches.is_empty());
        assert_eq!(batches.type_count(), 0);
    }

    struct DummyKind(PrimitiveTypeId);
    impl RenderPrimitive for DummyKind {
        fn type_key(&self) -> PrimitiveTypeId {
            self.0
        }
        fn build(&self, _ctx: &PipelineContext<'_>) -> Box<dyn Any + Send + Sync> {
            Box::new(())
        }
        fn draw(&self, _state: &mut (dyn Any + Send + Sync), _batch: BatchDraw<'_, '_>) {}
    }

    #[test]
    fn registry_register_and_lookup() {
        let mut registry = PrimitiveRegistry::new();
        assert!(!registry.contains(A));
        registry.register(Box::new(DummyKind(A)));
        registry.register(Box::new(DummyKind(B)));
        assert!(registry.contains(A));
        assert!(registry.contains(B));
        assert_eq!(registry.registered_type_count(), 2);
        // Re-registering the same type id replaces, not duplicates.
        registry.register(Box::new(DummyKind(A)));
        assert_eq!(registry.registered_type_count(), 2);
    }
}
