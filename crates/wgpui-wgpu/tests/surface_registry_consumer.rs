//! §8's Phase 4 gate, second clause: `SurfaceRegistry`'s existing concurrency
//! behaviour is unaffected by the consumer-side unification.
//! See docs/gpu-native-architecture.md §5.5, §9's risk table.
//!
//! §9 names the specific way this phase could go wrong — "unifying
//! `WgpuSurface`'s composite path with `.boundary()`'s accidentally touches
//! `SurfaceRegistry`'s cross-thread producer-side synchronization … hard-won,
//! carefully-documented concurrency code that has nothing to do with the bug
//! being fixed" — and says the existing tests are the gate, "not a new test
//! suite reverse-engineered from the concurrency doc comments."
//!
//! So there are three layers of evidence here, in increasing strength:
//!
//! 1. **The six existing model tests pass unmodified**, in
//!    `render/surface_registry.rs`'s own `mod tests`. They came across with the
//!    file and were not touched; `cargo test -p wgpui-wgpu --lib` runs them.
//! 2. **A differential**, below: the same producer script driven through the
//!    legacy consumer sequence and through the unified one produces
//!    *identical* observable registry state, step for step. This is the direct
//!    form of "unaffected" — not an argument that the code was not edited, but
//!    a measurement that its behaviour did not change.
//! 3. **A real cross-thread run**, below: a producer thread presenting at its
//!    own pace against a consumer compositing through `plan_composites`, with
//!    backpressure and pacing asserted rather than assumed.
//!
//! # What the legacy consumer sequence is
//!
//! Two calls, in this order, from `renderer.rs`'s `PrimitiveBatch::Surfaces`
//! arm: `swap_ready_display_if_new(id)` then `front_view(id)`. That is the
//! whole of the consumer's contact with the registry, and
//! `CompositeConsumer::view` makes the same two calls in the same order. The
//! differential below is what makes that a fact rather than a claim about a
//! diff.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use wgpui_core::boundary::compositor::{CompositeEntry, CompositeSource, ExternalSurfaceId};
use wgpui_core::geometry::Rect;
use wgpui_wgpu::render::device::context_or_report;
use wgpui_wgpu::render::pipelines::{CompositePipeline, TARGET_FORMAT};
use wgpui_wgpu::render::surface_registry::{SurfaceId, SurfaceRegistry};
use wgpui_wgpu::render::textures::external_surface::{CompositeConsumer, plan_composites};

/// What a step of a producer/consumer script does.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Step {
    /// The external render thread presents a frame.
    Produce,
    /// The compositor draws the surface.
    Composite,
    /// The compositor runs a frame in which the surface is not drawn.
    Skip,
}

/// The registry state a consumer can observe, sampled after every step.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Observation {
    generation: Option<u64>,
    unconsumed: bool,
}

fn observe(registry: &SurfaceRegistry, surface: SurfaceId) -> Observation {
    Observation {
        generation: registry.frame_generation(surface),
        unconsumed: registry.has_unconsumed_frame(surface),
    }
}

/// A script with every interleaving that matters: a paired 1:1 run, a producer
/// that outruns the consumer, a consumer that paints with nothing new, and
/// frames where the surface is not drawn at all.
fn script() -> Vec<Step> {
    vec![
        Step::Produce,
        Step::Composite,
        Step::Composite,
        Step::Produce,
        Step::Produce,
        Step::Composite,
        Step::Produce,
        Step::Skip,
        Step::Skip,
        Step::Composite,
        Step::Produce,
        Step::Composite,
        Step::Skip,
    ]
}

fn window() -> Rect {
    Rect::from_origin_size([0.0, 0.0], [512.0, 512.0])
}

/// **The differential**: identical observable registry state, step for step,
/// between the legacy consumer sequence and the unified one.
#[test]
fn the_unified_consumer_leaves_the_registry_in_the_same_state_as_the_legacy_one() {
    let Some(context) = context_or_report("surface_registry_differential") else {
        return;
    };
    let pipeline = CompositePipeline::new(&context.device);

    // Arm 1: the legacy sequence, spelled out here exactly as
    // `renderer.rs`'s surfaces batch spells it.
    let legacy = SurfaceRegistry::new();
    let legacy_surface = legacy.create(&context.device, 64, 64, TARGET_FORMAT);
    let mut legacy_trace = Vec::new();
    for step in script() {
        match step {
            Step::Produce => legacy.swap_rendering_ready_no_sync(legacy_surface),
            Step::Composite => {
                legacy.swap_ready_display_if_new(legacy_surface);
                let view = legacy.front_view(legacy_surface);
                assert!(view.is_some(), "a created surface always has a front view");
            }
            Step::Skip => {}
        }
        legacy_trace.push(observe(&legacy, legacy_surface));
    }

    // Arm 2: the unified consumer, through the same entry type a boundary
    // texture would take.
    let unified = SurfaceRegistry::new();
    let unified_surface = unified.create(&context.device, 64, 64, TARGET_FORMAT);
    let entry = CompositeEntry::sampled(
        CompositeSource::External(ExternalSurfaceId::from_raw(unified_surface.as_raw())),
        Rect::from_origin_size([0.0, 0.0], [64.0, 64.0]),
        window(),
    );
    let consumer = CompositeConsumer {
        registry: Some(&unified),
        textures: None,
    };
    let mut unified_trace = Vec::new();
    for step in script() {
        match step {
            Step::Produce => unified.swap_rendering_ready_no_sync(unified_surface),
            Step::Composite => {
                let plan = plan_composites(
                    &context.device,
                    &context.queue,
                    &pipeline,
                    &consumer,
                    std::slice::from_ref(&entry),
                );
                assert_eq!(plan.prepared.len(), 1, "an uncovered entry is drawn");
                assert_eq!(plan.culled, 0);
                assert_eq!(plan.unavailable, 0);
            }
            // A frame where the surface is not in the composite list at all —
            // the shape a frame takes when the panel is scrolled out of view.
            Step::Skip => {
                let plan =
                    plan_composites(&context.device, &context.queue, &pipeline, &consumer, &[]);
                assert!(plan.prepared.is_empty());
            }
        }
        unified_trace.push(observe(&unified, unified_surface));
    }

    assert_eq!(
        legacy_trace, unified_trace,
        "the unified consumer must leave the producer's state exactly where the \
         legacy one does — §9's risk, measured rather than argued"
    );
    assert!(
        legacy_trace.iter().any(|state| state.unconsumed),
        "the script must actually reach a backpressure state, or it proves nothing"
    );
    assert!(
        legacy_trace.iter().any(|state| !state.unconsumed),
        "and must actually clear it again"
    );
    println!(
        "surface_registry_differential: {} steps identical",
        legacy_trace.len()
    );
}

/// **The one behavioural difference, asserted so it is not mistaken for a
/// regression.** A composite entry the layer tier culls does not reach the
/// registry at all, so a produced frame stays unconsumed — which is §5.5's
/// promised win, not a broken consumer.
#[test]
fn a_culled_entry_leaves_the_producers_frame_unconsumed() {
    let Some(context) = context_or_report("surface_registry_culled") else {
        return;
    };
    let pipeline = CompositePipeline::new(&context.device);
    let registry = SurfaceRegistry::new();
    let surface = registry.create(&context.device, 64, 64, TARGET_FORMAT);
    registry.swap_rendering_ready_no_sync(surface);

    let viewport = CompositeEntry::sampled(
        CompositeSource::External(ExternalSurfaceId::from_raw(surface.as_raw())),
        Rect::from_origin_size([100.0, 100.0], [200.0, 200.0]),
        window(),
    );
    let cover = CompositeEntry {
        source_is_opaque: true,
        ..CompositeEntry::sampled(
            CompositeSource::External(ExternalSurfaceId::from_raw(surface.as_raw() + 1000)),
            window(),
            window(),
        )
    };

    let before = observe(&registry, surface);
    let plan = plan_composites(
        &context.device,
        &context.queue,
        &pipeline,
        &CompositeConsumer {
            registry: Some(&registry),
            textures: None,
        },
        &[viewport, cover],
    );
    assert_eq!(plan.culled, 1);
    assert_eq!(
        observe(&registry, surface),
        before,
        "a culled entry must not touch the registry at all — not the generation, \
         not the composited generation, nothing"
    );
    assert!(registry.has_unconsumed_frame(surface));
}

/// **A real cross-thread run.** A producer thread presenting at its own pace
/// against the unified consumer, with the two backpressure properties
/// `WgpuSurfaceHandle`'s documentation states asserted rather than assumed.
#[test]
fn a_producer_thread_paces_itself_against_the_unified_consumer() {
    let Some(context) = context_or_report("surface_registry_threaded") else {
        return;
    };
    let pipeline = CompositePipeline::new(&context.device);
    let registry = Arc::new(SurfaceRegistry::new());
    let surface = registry.create(&context.device, 64, 64, TARGET_FORMAT);

    let stop = Arc::new(AtomicBool::new(false));
    let produced = Arc::new(AtomicU64::new(0));
    let skipped_for_backpressure = Arc::new(AtomicU64::new(0));

    let producer = std::thread::spawn({
        let registry = Arc::clone(&registry);
        let stop = Arc::clone(&stop);
        let produced = Arc::clone(&produced);
        let skipped = Arc::clone(&skipped_for_backpressure);
        move || {
            // Exactly the loop `WgpuSurfaceHandle::has_unconsumed_frame`'s doc
            // prescribes: "skip producing while it returns true."
            while !stop.load(Ordering::Relaxed) {
                if registry.has_unconsumed_frame(surface) {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    std::thread::yield_now();
                    continue;
                }
                registry.swap_rendering_ready_no_sync(surface);
                produced.fetch_add(1, Ordering::Relaxed);
            }
        }
    });

    let entry = CompositeEntry::sampled(
        CompositeSource::External(ExternalSurfaceId::from_raw(surface.as_raw())),
        Rect::from_origin_size([0.0, 0.0], [64.0, 64.0]),
        window(),
    );
    let consumer = CompositeConsumer {
        registry: Some(&registry),
        textures: None,
    };

    let started = Instant::now();
    let mut composited = 0u64;
    let mut last_generation = 0u64;
    while composited < 200 && started.elapsed() < Duration::from_secs(10) {
        let plan = plan_composites(
            &context.device,
            &context.queue,
            &pipeline,
            &consumer,
            std::slice::from_ref(&entry),
        );
        assert_eq!(plan.prepared.len(), 1);
        composited += 1;
        let generation = registry
            .frame_generation(surface)
            .expect("the surface is registered");
        assert!(
            generation >= last_generation,
            "the producer generation must never go backwards: {generation} \
             after {last_generation}"
        );
        last_generation = generation;
    }
    stop.store(true, Ordering::Relaxed);
    producer.join().expect("the producer thread must not panic");

    let produced = produced.load(Ordering::Relaxed);
    let skipped = skipped_for_backpressure.load(Ordering::Relaxed);
    println!(
        "surface_registry_threaded: {composited} composites, {produced} frames \
         produced, {skipped} production attempts skipped for backpressure"
    );

    assert!(
        composited >= 200,
        "the consumer stalled: only {composited} composites in 10s, which means \
         the unified consumer blocked on the producer"
    );
    assert!(
        produced > 0,
        "the producer never presented, so nothing about pacing was tested"
    );
    assert!(
        skipped > 0,
        "backpressure never engaged, so the property the producer's own \
         documentation prescribes was never exercised"
    );
    assert!(
        !registry.has_unconsumed_frame(surface) || produced > 0,
        "a surface with an unconsumed frame must have had one produced"
    );
}

/// The producer API this phase must not have touched, checked as a compile-time
/// fact rather than by reading the diff: every method the external render thread
/// uses is still callable with the same signature.
#[test]
fn the_producer_side_api_is_unchanged() {
    let Some(context) = context_or_report("surface_registry_producer_api") else {
        return;
    };
    let registry = SurfaceRegistry::new();
    let surface = registry.create(&context.device, 32, 32, TARGET_FORMAT);

    // The five calls an external render thread makes, in the order
    // `WgpuSurfaceHandle`'s doc example makes them.
    let (view, (width, height)) = registry
        .lock_and_get_back_with_size(surface)
        .expect("a created surface has a back buffer");
    assert_eq!((width, height), (32, 32));
    drop(view);
    assert!(registry.back_view(surface).is_some());
    registry.swap_rendering_ready_no_sync(surface);
    assert!(
        !registry.set_redraw_pending(surface),
        "the flag was not already set, so this call is the one that sets it"
    );
    assert_eq!(registry.get_pending_surfaces(), vec![surface]);
    registry.clear_redraw_pending(surface);
    assert!(registry.get_pending_surfaces().is_empty());
    assert_eq!(registry.size(surface), Some((32, 32)));
    assert_eq!(registry.format(surface), Some(TARGET_FORMAT));
    assert!(registry.resize(&context.device, surface, 48, 48));
    assert_eq!(registry.size(surface), Some((48, 48)));
    registry.remove(surface);
    assert_eq!(registry.frame_generation(surface), None);
}
