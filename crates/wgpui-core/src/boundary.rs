//! `.boundary()`: the single cache-boundary primitive, a pure compositing
//! and buffering policy layered on top of always-on reconciliation.
//! See docs/gpu-native-architecture.md §4.1.
//!
//! # What a boundary is, and what it deliberately is not
//!
//! It answers "does this region get its own GPU texture, an overdraw margin,
//! and its own occlusion tier," never "does this region's content get diffed."
//! §4.0 answered the second question yes, for everything, in Phase 1, and
//! nothing in this module can change that answer: no type here is reachable
//! from `reconcile/reconciler.rs`'s reuse decision, and the reconciler's own
//! signature has no boundary parameter to fence it with. Phase 2's second gate
//! is precisely a test that removing a `.boundary()` costs the compositing
//! optimization and nothing else.
//!
//! Three files, three concerns: [`policy`] is what an author may tune,
//! [`identity`] is how a boundary finds itself across frames without being
//! named, and [`compositor`] is the per-frame decision that consumes both plus
//! [`crate::invalidation::reason::Reason`].

pub mod compositor;
pub mod identity;
pub mod policy;

pub use compositor::{BoundaryComposite, BoundaryState, Composite, Compositor, TiledVisit};
pub use identity::BoundaryIdentity;
pub use policy::{BoundaryPolicy, Buffering, Pixels, Retention, Size};
