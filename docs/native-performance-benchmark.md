# Native performance benchmark

`native_performance_bench` is the reproducible native baseline for the retained
pipeline. It creates a real `winit` window and `wgpu::Surface`, warms each
case, presents timed frames, and writes one machine-readable JSON report.

Run it with the optional diagnostics feature:

```text
cargo run -p wgpui-wgpu --release --features devtools --example native_performance_bench -- --diagnostics both
```

Useful smaller runs for local iteration are:

```text
cargo run -p wgpui-wgpu --example native_performance_bench -- --siblings 8 --depth 2 --warmup 1 --frames 3 --diagnostics off
```

Each report contains `steady`, `scroll`, and `continuous` cases for every
requested diagnostics setting. The workload has N stable siblings, configurable
nesting depth, a bounded scroll boundary, rounded surfaces with blurred
shadows, shaped raw text, and one surface whose display value changes every
frame. The timed stages are `description_build`, `reconciliation`, `layout`,
`shared_walk`, `emission`, `damage`, `uploads`, `visibility`, and `present`.

`description_build` includes fresh `Description` construction and the frame
loop's raw-text materialization. `shared_walk` is the existing retained-plan
interaction walk. `damage` is patch application, dirty-layer selection, and
diagnostic-region preparation. `visibility` is the existing dirty-layer
ordering/occlusion CPU dispatch timing returned by `FrameRenderer`; the tiled
visibility dispatch remains measured by `phase45_tiling_bench` because the
current `FrameLoop` does not route tiled boundaries through that pass.
`uploads` is the existing scene-arena upload timing; glyph-atlas synchronization
has no separate timing hook today.

The JSON `cost_centers` list is sorted by total timed nanoseconds and is the
machine-readable answer for continuous invalidation cost centers. The counters
show whether the continuously changing surface re-emits only its own records,
how many scene bytes are uploaded, and whether steady/scroll frames remain
resident. Warmups are excluded from stage summaries and diagnostics snapshots.

The benchmark is intentionally a capture path: normal `FrameLoop::draw` does
not collect these stage clocks, and the devtools registry is disabled unless
`WGPUI_RENDER_STATS` or the benchmark's explicit A/B setting enables it. Native
present timing includes the configured surface present mode's pacing. Results
should therefore be compared on the same adapter, power state, display mode,
viewport, build profile, and diagnostics setting; software adapters are marked
in the report and are not hardware evidence.
