# Phase 7 results

Phase 7 establishes the devtools extraction boundary on the post-Phase-6.6
`2.0` branch. `wgpui-core` owns the backend-neutral
`InstrumentationHooks` contract (`begin_span`/`end_span`, counters, frame
completion, and optional GPU timestamp pairs). `wgpui-wgpu` can enable the
optional `devtools` feature and reports frame CPU spans and successful frame
completion through that contract. With the feature disabled, neither core nor
the backend has a devtools dependency.

`wgpui-devtools` now has clean `flamegraph`, `render_stats`, `inspector`, and
`perf_ab_tests` modules with crate-root exports for the portable records and
the `DevtoolsHooks` implementation. Render statistics preserve the legacy
environment gate and snapshot/reset behavior for the extracted portion.

## Coverage

- Core hook span construction is tested, including the no-op implementation
  contract.
- Devtools render-stat disabled behavior is tested; the implementation retains
  named counters, timer samples, snapshots, reset, and RAII scopes.
- `cargo test -p wgpui-core` and `cargo test -p wgpui-devtools` cover the
  portable hook and stats paths.
- `cargo check --workspace` covers the feature-off dependency graph; a separate
  `wgpui-wgpu` check with `--features devtools` covers the feature-on graph.

## Bounded legacy adapters and remaining work

The full legacy `src/flamegraph.rs`, `src/flamegraph_gpu.rs`,
`src/flamegraph_replay.rs`, `src/flamegraph_ui_capture.rs`, and
`src/inspector.rs` cannot be moved wholesale yet. They depend on private
legacy `gpui` types and renderer state: `App`/`Window`/`Element`, legacy
`Scene` and atlas IDs, private GPU query-manager methods, and the legacy
shader/texture layout. Copying those files into the new crate would either
create a dependency cycle or make a fake API that cannot execute. Phase 7
therefore exposes bounded data/adapter seams (`ElementRecord`,
`ReplayViewport`, `record_timestamp`, and `CaptureRequest`) and leaves the
legacy implementation in place. Phase 8 must implement the real frontend
adapter and move replay/readback only after the new scene and renderer types
have stable public contracts; the compatibility façade is intentionally not
part of this phase.
