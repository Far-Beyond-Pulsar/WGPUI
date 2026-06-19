# WGPUI Claude Instructions

## Mission

WGPUI is a single-crate `gpui-ce` fork that keeps GPUI's public API familiar while replacing the per-platform rendering and windowing stack with one `wgpu` + `winit` path. Preserve that direction: do not reintroduce split Metal/Vulkan/D3D/Cocoa/Win32/Wayland architecture unless the user explicitly asks for a compatibility shim.

The near-term work is rendering-performance porting from the older multi-crate WGPUI prototype into this single-crate Pulsar/WGPUI tree. Keep behavior correct first. Performance wins are not done until they are validated with real rendering, scrolling, and display-backed examples.

## Current Rendering-Perf State

Use `RENDERING_PERF_PORT_PLAN.md` as the live source of truth. Current committed phases are:

- Phase 1 — scene chunk tracking and incremental sort.
- Phase 2 — scene-side damage computation, currently dormant.
- Phase 4 — per-view layout caching, active but not display-validated.
- Phase 6 — scene batch counting metric.
- Phase 0 — frame metrics collection.

Remaining work must not be blind-ported as a mechanical patch:

- Phase 5 list/uniform element caching must be co-ported with display validation. Pulsar's `list.rs` has divergent `layout_all_items`, `layout_items`, and `smooth_scroll` paths, and stale element-cache handle propagation can cause missing or stale transcript rows.
- Phase 3 GPU diff-upload needs generation semantics reworked. If `scene.generation` changes every frame, generation-skip logic is effectively a no-op.
- Phase 7 buffer compaction depends on Phase 3's persistent buffer model.

Before Phase 5, Phase 3, or Phase 7 work, run an inspector or example on a real display against the current base and verify drawing plus scrolling behavior.

## Architecture First

Every change should have an explicit layer:

- API/entity layer: public GPUI types, `Entity<T>`, `Context<T>`, `App`, `Window`, actions, events.
- Element/layout layer: `Render`, `RenderOnce`, element cache, list/uniform list behavior, layout reuse.
- Scene layer: primitive collection, scene chunks, sorting, damage, metrics.
- Renderer layer: `src/platform/cross/`, `wgpu` resources, atlases, batches, upload/draw scheduling.
- Windowing/runtime layer: `winit`, frame requests, invalidation, presentation, input.
- Tests/examples/docs layer: reproducible behavior checks, display-backed repros, and written handoff notes.

Do not build isolated features that only work in one example. A feature should compose with the same entity, element, scene, renderer, and validation model as the rest of the crate.

Avoid god objects. Do not let one `Window`, renderer, context, cache, or metrics struct absorb unrelated routing, extraction, layout, GPU upload, debugging, and policy state. Keep ownership boundaries explicit.

## Working Rules

- Treat `research` notes, upstream forks, and old multi-crate WGPUI commits as reference material, not as code to paste blindly.
- Prefer editing existing files when the change belongs to an existing subsystem.
- Do not create `mod.rs` files.
- Avoid `unwrap()` and panicking indexing in new code; propagate errors or handle them explicitly.
- Never silently discard fallible errors with `let _ =`; propagate, log, or match them.
- Use full words for variable names.
- Use variable shadowing to scope clones moved into async tasks.
- In entity update closures, use the inner `cx`, not an outer context.
- Avoid updating an entity while it is already being updated.
- Use current GPUI APIs: `Entity<T>`, `App`, `Context<T>`, explicit `Window`, and async-closure `spawn` calls.
- Do not use removed APIs such as `Model`, `View`, `AppContext`, `ModelContext`, `WindowContext`, or `ViewContext`.

## Validation

For Rust source changes, prefer this sequence:

```sh
cargo fmt --all
./script/clippy
cargo test --workspace --all-targets
```

Use `./script/clippy` instead of `cargo clippy`.

For rendering or windowing changes, also run the smallest relevant example on a display. Existing examples include:

```sh
cargo run --example karaoke_app
cargo run --example karaoke_text
cargo run --example text_gradients
```

When a display is unavailable, say so directly and do not mark display-sensitive work as fully validated.

## Project Claude Layout

- `CLAUDE.md` is the primary project instruction file.
- `AGENTS.md` exists for agents that do not read `CLAUDE.md`; keep it compatible and concise.
- `MEMORY.md` is local scratch memory and is ignored by git.
- `.claude/agents/` contains focused helper-agent instructions.
- `.claude/commands/` contains repeatable command workflows.
- `.claude/rules/` contains reusable local rule fragments.
- `.claude/skills/` contains project-local skills only when they are genuinely useful.