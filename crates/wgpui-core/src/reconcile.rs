//! Ambient reconciliation: `ElementInstance`/`InstanceKey` diffing that
//! applies to every element in the window, not fenced to a `.boundary()`
//! subtree. See docs/gpu-native-architecture.md §4.0, constraint 5 (§0).
//!
//! Constraint 5 states the inversion this module exists to make real: "the
//! default assumption, everywhere, is *retained unless a diff proves
//! otherwise* — never *rebuilt unless something opted into caching*." The
//! legacy backend built every mechanism this needs and then fenced it to
//! `.layer()` subtrees, which SFD §0.1 measured as producing near-zero real
//! adoption (1 of 37 call sites). Nothing here has a fence to remove because
//! nothing here has one to begin with.
//!
//! # Module map
//!
//! | Module | Role | §3.1 |
//! |---|---|---|
//! | [`description`] | the cheap per-frame value | addition |
//! | [`diff_key`] | the fingerprint trait and its comparison rules | mapped |
//! | [`instance`] | the retained record and the window-wide table | mapped |
//! | [`plan`] | what a frame's reconciliation decided, as data | addition |
//! | [`reconciler`] | the walk | addition |
//! | [`state`] | element state, which `.uncached()` must not touch | addition |
//! | [`uncached`] | the scope flag (§4.2) | mapped |
//!
//! Every addition is recorded, with its reasoning, in
//! `docs/phase-1-results.md` and in the module's own doc comment.

pub mod description;
pub mod diff_key;
pub mod instance;
pub mod plan;
pub mod reconciler;
pub mod state;
pub mod uncached;
pub mod walk;

pub use description::{Description, ElementId};
pub use diff_key::{AlwaysDirty, ReconcileKey, compare_by_equality};
pub use instance::{ElementInstance, InstanceKey, InstanceTable, RetainedElement};
pub use plan::{FramePlan, FrameStats, NodeOutcome, PlannedNode, RebuildReason};
pub use reconciler::{ReconcileError, Reconciler};
pub use state::{ElementStateStore, StateKey, StateScope};
pub use uncached::UncachedScope;
pub use walk::{WalkNode, shared_walk};
