# WGPUI 2.0 release-candidate audit

Audit date: 2026-08-30
Scope: workspace-native crates and the declared `wgpui-examples-2` examples. This was a read-only implementation audit; only this document is intended to change.

## Gate result

**BLOCKING.** The native production libraries pass a locked offline build, but the release candidate is not releasable: the focused WGPU test suite has one shader/protocol failure, core test-support code does not compile after the `Quad` shape gained `material`, and the consolidated example analyzer reports 45/45 failures with 67 unique normalized errors.

## Evidence

| Check | Result | Classification |
|---|---|---|
| `cargo metadata --locked --format-version 1 --no-deps` | Passed; 12 workspace packages | Fixed |
| `cargo check --workspace --locked --offline` | Production libraries compile; Cargo emits the existing workspace resolver/profile warnings | Fixed for build; warning cleanup remains non-blocking |
| Isolated cold `cargo check --workspace --locked --offline --all-targets` | Failed in `wgpui-examples-2` | Blocking |
| `cargo test -p wgpui-core --locked --offline --lib` | 375 tests discovered, compilation fails in test fixtures with 7 missing `Quad.material` fields | Blocking |
| `cargo test -p wgpui-wgpu --locked --offline --lib` | 77 passed, 1 failed | Blocking |
| `cargo test -p wgpui-layout --locked --offline --lib` | 6 passed | Fixed |
| `script/analyze-examples-2-errors.ps1 -Locked -Offline` | 45 total, 0 passed, 45 failed, 67 unique errors | Blocking |
| `script/clippy.ps1 -p wgpui-core --locked --offline` | Blocked by the same missing `Quad.material` test fixtures before a warning verdict | Blocking |

The cold builds used isolated directories under `target/rc-audit-*`; no existing build artifacts were removed. Offline resolution/building succeeded, so this audit observed no download required from an uncached registry or git source. A checked-in approved dependency-graph baseline was not found, so “no new package/feature/transitive download” cannot be proven by graph comparison.

## Findings

### Blocking

1. **Example/API compatibility is not closed.** The analyzer fails every declared example. Representative failures include `Render::render` signature mismatches, missing `canvas`, `uniform_list`, `vlist`, `WgpuSurfaceHandle`, `TextOverflow`, and `PathBuilder` exports, changed `App`/`Entity` APIs, missing blur/focus/refresh methods, and incompatible geometry types. The full machine-readable evidence is `target/rc-audit-examples/report.json`; the consolidated report is `target/rc-audit-examples/report.md`. The gate requires every declared example to compile or have an explicit, justified exclusion.

2. **Core test-support fixtures are stale.** `crates/wgpui-core/src/test_support/ui_walk.rs`, `src/occlusion.rs`, `src/patch/apply.rs`, `src/patch/primitive.rs`, and `src/test_support/raster.rs` construct `Quad` without its required `material` field. This prevents the core test library, widget tests, and strict clippy target from compiling. Implementation files were not edited under this audit.

3. **WGPU shader/protocol differential fails.** `render::pipelines::tests::the_shaders_agree_with_the_protocol_about_a_quad_slot` fails at `crates/wgpui-wgpu/src/render/pipelines.rs:1257`, observing 144 bytes versus the protocol’s 80 bytes. The other 77 WGPU library tests pass, including capability and retention tests, but this byte-layout mismatch blocks the renderer gate.

4. **Production shaping errors are discarded.** `crates/wgpui-widgets/src/styled_text.rs:672-677` deliberately ignores the production error after test-only panic behavior. This violates the RC requirement that async/operation failures reach the UI with meaningful feedback and makes a failed text emission silent. It is separate from the test panic inventory.

5. **Dependency comparison is incomplete.** `Cargo.lock` is locked and offline-resolvable, with pinned Helio, Quark, and priority-threadpool git revisions, but there is no approved final graph artifact in the repository against which new packages, features, target edges, build scripts, and transitive downloads can be mechanically compared. Record that baseline before release sign-off.

### Intentional

1. **Scoped placeholder modules are documented, not empty accidents.** `wgpui-layout` explicitly marks `containment.rs` and `regular.rs` as Phase 0 stubs; `wgpui-widgets` documents the remaining event/interaction/scroll seams and the description-only `wgpu_surface` boundary. Their files contain contracts or state-machine code where applicable, and the production build does not depend on an unimplemented panic path. These are intentional scope exclusions, but must remain named in release notes if the corresponding APIs are advertised.

2. **Invariant panics are test/build guards, not ordinary input handling.** The production scan found compatibility task double-polling/entity-initialization invariants, reconciliation invariants, and build-script manifest/lockfile guards. The many `expect` calls in raster, atlas, and differential code are test-only or assert locally constructed valid fixtures. The audit does not classify these as ordinary production failure paths; fuzz/error-path coverage should still prevent externally supplied malformed data from reaching them.

### Fixed / verified

1. **GPU capability negotiation is capability-driven.** `wgpui-wgpu/src/render/device.rs` requests the adapter with a compatible surface when available, intersects optional indirect features with `adapter.features()`, and uses `adapter.limits()` rather than hardcoded limits. The renderer derives its actual indirect support from granted device features. This addresses the previously risky hard-required-feature pattern.

2. **Surface configuration checks capabilities.** The WGPU window tests cover rejection of empty present modes and missing readback usage, and the implementation chooses a supported configuration rather than assuming one present mode. This is fixed for the reviewed surface path.

3. **Retention mechanisms have focused coverage.** Passing core test output before fixture compilation failure includes unchanged-tree reuse, invalidation-axis behavior, delta uploads, layer/slab release, atlas eviction, tiled panning, and boundary retention tests. Passing WGPU tests include steady-state allocation behavior, boundary texture retention, surface producer/compositor behavior, atlas eviction, and capability validation. These provide useful evidence but cannot close the overall gate while the fixture and shader failures remain.

4. **Shader files are non-empty except the explicitly deferred kernel.** The native shader inventory contains 13 tracked WGSL files with non-zero sizes. `wgpui-core/src/shaders/layout_uniform.wgsl` is the one explicitly documented placeholder for the deferred regular-content GPU layout kernel; the other renderer shaders contain implementation text. The placeholder is intentional only if that kernel is outside the RC feature claim.

## Final gate checklist

- [ ] Add `Quad.material` to every affected test-support fixture, then run core, widget, text, compatibility, and strict clippy checks.
- [ ] Correct the quad shader/protocol byte-layout mismatch and rerun all 78 WGPU library tests plus relevant integration differentials.
- [ ] Make all 45 declared examples compile, or remove/mark each unsupported example with a documented platform/API reason and an explicit gate decision; rerun the consolidated analyzer.
- [ ] Replace silent production shaping-error discard with an observable UI-layer error path and add a regression test.
- [ ] Capture and check in the approved dependency graph/baseline; compare package, feature, target-edge, build-script, git-revision, and download deltas.
- [ ] Run `cargo metadata --locked --offline`, isolated cold `cargo check --workspace --locked --offline --all-targets`, focused tests, release tests, and `script/clippy.ps1 --locked --offline` with `--deny warnings`.
- [ ] Validate every WGSL module on each available backend, including bindings, alignment, interpolation, storage usage, and feature guards.
- [ ] Exercise native adapter capabilities, device-loss/error behavior, resize/present paths, and the CPU indirect-draw fallback on hardware beyond the current Windows adapter.
- [ ] Complete retention stress coverage: panning in both axes, reveal/refill, LRU eviction, oversized content, scale changes, boundary destruction, repeated create/drop cycles, and resource-lifetime assertions.
- [ ] Re-run legacy/2.0 differential coverage for mixed primitives, scaled images, clipping, opacity, gradients, paths, blur, text, shadows, underlines, surfaces, and empty/degenerate/NaN/Infinity inputs.
- [ ] Verify the final worktree contains only the intended audit document change before tagging the release candidate.
