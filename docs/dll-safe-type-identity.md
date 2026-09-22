# Cross-DLL State Safety for the Plugin Boundary

Status: proposal.

This proposes eliminating an entire class of bug we've hit twice in one
debugging session (script-editor-plugin crashes, 2026-09-22) and will keep
hitting indefinitely under the current architecture: state that the framework
assumes is process-singleton, but which is actually duplicated once per
statically-linked binary. Two independent mechanisms close it —
`PortableTypeId` for type identity, and a real OS-level TLS slot for the
handful of genuinely-ambient "current X" pointers — and neither requires an
API change visible to plugin authors or a build-system change.

---

## 0. Root cause

`gpui-ce` (`crates/ui/wgpui`) has no `crate-type` set, so it's an rlib. An
rlib is not a runtime artifact — it's compiled object code that gets **copied
into whatever links it, at compile time**. `pulsar_engine.exe` links it and
gets a copy baked in. Every plugin (`plugins/vendor/*`, each built with
`crate-type = ["cdylib", "rlib"]` and depending on `gpui-ce` by path — see
`plugins/vendor/code_editor/Cargo.toml:23`) links it too, and gets its own
**separate** copy baked in, because each plugin is its own independent
`cargo build`, not part of `pulsar_engine`'s workspace build graph.

There is no single "the framework" at runtime. There are N of them — one per
binary — that happen to have been compiled from identical source. Two
consequences follow, and they are different bugs with different fixes:

**Every `thread_local!`/`static` in `gpui-ce` is duplicated N times.**
Rust's `thread_local!` macro expands independently in each compiled crate
instance; each instance allocates its own storage. `CURRENT_ELEMENT_ARENA`
(`src/window.rs:928`) is a thread-local pointer to "the arena currently
active for this draw." When the host enters it (`Window::draw`,
`src/window.rs:3350`) and a plugin's `PanelView::title()` later constructs an
element, the plugin's own compiled copy of `with_element_arena`
(`src/window.rs:947`) checks the **plugin's own, never-set** copy of that
thread-local and panics: `element arena not active`. This crashed the script
editor today the moment its tab got a title bar
(`crates/ui/wgpui-component/crates/ui/src/dock/tab_panel/render.rs:235`
calling into `plugins/vendor/code_editor/src/script_editor/mod.rs:544`).

**`std::any::TypeId` is not guaranteed stable across separate compilations**,
even of identical source with identical flags — this is documented Rust
behavior, not a bug in our build. `ActionRegistry` is built once, at `App`
startup, from `inventory::iter()` (`src/action.rs:290`), which only sees
`actions!`/`#[action]` macro registrations statically linked into the host
binary at that point. `plugins/vendor/code_editor/src/script_editor/mod.rs:26`
defines `actions!(script_editor, [SaveCurrentFile, CloseCurrentFile])` inside
the plugin — the host's registry has never heard of these. The first time one
was dispatched, `ActionRegistry::discriminator_for_type` (now removed, was
`src/action.rs:347`) called `.expect("action type not registered")` on a
`TypeId` the registry never indexed, and panicked.

Both bugs were fixed today by patching the specific call site
(`src/window.rs:1281`'s `action_instance_discriminator`, and
`crates/ui/wgpui-component/crates/ui/src/dock/panel.rs:255`'s
`ElementArenaScope::enter`). Both fixes are correct and should stay. Neither
is the real fix.

---

## 1. Why patching call sites doesn't converge

A survey of the current tree:

```
$ grep -rln 'TypeId'        src/ --include=*.rs | wc -l   # 10 files
$ grep -rn  'TypeId::of::<' src/ --include=*.rs | wc -l   # 60 sites
$ grep -rln 'thread_local!' src/ --include=*.rs | wc -l   #  5 files, 7 macros
```

Every one of those 60 `TypeId` sites and 7 thread-locals is a place someone,
at some point, reached for the ergonomic tool Rust hands you by default —
and every one of them is a latent instance of the same bug, waiting for a
plugin to exercise the code path that reaches it. §3 and §4 categorize all of
them. The pattern in both bugs fixed today was identical: find the crash,
find the one call site, patch it. That works exactly once per call site. It
does not work for the 58 sites we haven't crashed on yet, and it does not
work for whatever a plugin author or contributor writes next month, because
there's no barrier stopping someone from writing `TypeId::of::<T>()` or
`thread_local!` again — it's still the obvious, ergonomic, *first* thing
Rust offers, and nothing marks it as unsafe in this codebase.

The fix has to remove the footgun, not keep finding people who stepped on it.

---

## 2. The fix

Two independent mechanisms, because the two bugs have different shapes.
`TypeId` divergence is about a *key* being wrong; the arena is about
*storage* being duplicated. Fixing one does not fix the other, and a full
dylib conversion — the obvious "just make it actually shared" answer — is
ruled out for a specific, structural reason (§6.1): `gpui-ce` is saturated
with generics (`Entity<T>`, `Context<T>`, `WeakEntity<T>`), and a Rust
`dylib` can only serve monomorphizations it already contains. It cannot hand
a plugin a fresh `Entity<ScriptEditor>` for a type it's never heard of. So
the framework itself stays exactly as it is — an rlib, recompiled into every
binary — and both fixes live *inside* it, invisible from outside.

### 2.1 `PortableTypeId` — name-hash identity, drop-in for `TypeId`

```rust
// src/portable_type_id.rs (new)

/// A type identifier that is stable across separately-compiled binaries
/// linking the same source, unlike `std::any::TypeId`. Computed from
/// `std::any::type_name::<T>()`, which is deterministic source text, not a
/// compiler-internal hash — the same reasoning already validated by
/// `action_name_hash`/`type_name_hash` (`src/window.rs:1267,1275`), which
/// this consolidates and supersedes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PortableTypeId(u64);

impl PortableTypeId {
    pub fn of<T: 'static + ?Sized>() -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::any::type_name::<T>().hash(&mut hasher);
        Self(hasher.finish())
    }

    /// For types whose identity is defined by a registered name rather than
    /// `type_name::<T>()` (actions, which the `#[action(name = "...")]`
    /// macro attribute can override) — hash the name directly instead.
    pub fn of_name(name: &str) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        name.hash(&mut hasher);
        Self(hasher.finish())
    }
}
```

This reads exactly like `TypeId::of::<T>()` at every call site — the swap is
`HashMap<TypeId, V>` → `HashMap<PortableTypeId, V>`, `TypeId::of::<T>()` →
`PortableTypeId::of::<T>()`. No plugin-facing signature changes anywhere:
`cx.on_action::<MyAction>(...)`, `Entity<T>`, `cx.emit(...)` all keep their
current call syntax. This is purely an internal representation swap inside
`gpui-ce`.

Collision note: a 64-bit hash of a globally-unique string has the same
collision profile `action_name_hash` already has in production today — not
new risk, just consolidated into one type instead of three ad hoc `fn`s
(`action_name_hash`, `type_name_hash`, and the instance-based
`action_instance_discriminator` added today all become one call site on
`PortableTypeId::of::<T>()`).

### 2.2 A real OS-level TLS slot for genuinely-ambient state

`with_element_arena` (`src/window.rs:947`) is called from `AnyElement::new`
(`src/element.rs:872`), which is reached from
`IntoElement::into_any_element(self)` — a signature with **no `cx`/`window`
parameter at all**. That's why this one thread-local exists in the first
place, and why `PortableTypeId` alone doesn't fix it: there's no map to key
into at that call site, because there's nothing to look anything up with.

The fix is to back it with a genuine OS-level TLS slot instead of Rust's
`thread_local!` macro. The distinction matters: `thread_local!` is a macro
that allocates new storage in *each compiled crate instance*; raw
`TlsAlloc`/`pthread_key_create` allocate a slot from the **OS's own
process-wide table**, identified by an integer index the OS hands back. If
that index is allocated exactly once — by the host, in `App::new()` — and
shared explicitly through the one struct every plugin already gets a real
pointer to (`App` itself), then every copy of `gpui-ce`'s code, host or
plugin, reads and writes the *same* OS-level slot by index. The OS doesn't
care how many times the crate's source was compiled.

```rust
// src/shared_tls.rs (new)

/// A TLS slot allocated once from the OS, identified by an index that can be
/// shared across independently-compiled binaries linking the same source —
/// unlike `std::thread_local!`, which allocates separate storage in each one.
///
/// Still genuinely thread-local (not a bare process-global): multiple `App`
/// instances can be interleaved on one thread by the test scheduler
/// (`src/window.rs:930`'s existing comment on `CURRENT_ELEMENT_ARENA`), and
/// this preserves that isolation — it's `thread_local!` specifically that's
/// wrong here, not thread-locality itself.
pub struct SharedTlsSlot(RawSlot);

impl SharedTlsSlot {
    /// Allocate a new OS TLS slot. Call exactly once per process (from
    /// `App::new()`) and share the resulting `SharedTlsSlot`/index via `App`
    /// — do not call this from plugin code or per-DLL init.
    pub fn alloc() -> Self { .. }

    /// Reconstruct a handle to an already-allocated slot from its raw index
    /// (crossed the FFI boundary as a plain integer field on `App`).
    pub fn from_raw(index: RawSlot) -> Self { .. }

    pub fn get(&self) -> Option<*const ()> { .. }
    pub fn set(&self, ptr: *const ()) { .. }
    pub fn raw(&self) -> RawSlot { self.0 }
}

// cfg-gated backend: TlsAlloc/TlsGetValue/TlsSetValue (windows-sys) on
// windows; pthread_key_create/pthread_getspecific/pthread_setspecific
// (libc) on unix. See §7 for the wasm target.
```

`App` gains one field: `element_arena_tls: SharedTlsSlot`, allocated in
`App::new()`. `ElementArenaScope::enter`/`with_element_arena`
(`src/window.rs:947,990`) change their backing store from
`CURRENT_ELEMENT_ARENA.with(...)` to `cx.element_arena_tls.get()/.set(...)`
— same call sites, same RAII shape, different storage underneath.

**This is the change that proves it's the real fix, not another patch**:
once the slot is genuinely process-wide, a plugin no longer needs to
re-enter the scope at its own generic-instantiation boundary — it's already
set, because it's the same OS slot the host set. That means:

- `view.rs:54`'s per-render `ElementArenaScope::enter` call becomes
  redundant (the top-level entry in `Window::draw` is the only one needed) —
  though it's cheap and harmless to leave as defense in depth.
- The `panel.rs:255,259` fix from today (§0) becomes **unnecessary** and
  should be reverted once this lands. If it isn't — if removing it
  reintroduces the crash — that means this design has a bug, and is the
  test for whether it actually worked.

---

## 3. Audit: every `TypeId` site in `gpui-ce` today

| Site | Keys | Plugin-reachable? | Tier |
|---|---|---|---|
| `entity_map.rs:364` `AnyEntity::downcast` | entity's concrete type | **Yes** — every `Entity<PluginType>` downcast | 1 — has an existing unsafe escape hatch (`lease_unchecked`/`read_unchecked`, `entity_map.rs:158,180`, doc'd "avoids TypeId checks for cross-DLL access") because this was already known unsafe |
| `window.rs:8097,8121` + `key_dispatch.rs:135,336` `DispatchActionListener.action_type` | action dispatch | Yes | 1 — partially fixed today (`window.rs:1281`); `global_action_listeners` (`app.rs:652`, `app.rs:2033` `App::on_action`) still keys on raw `TypeId` with no fallback — same bug, not yet hit |
| `action.rs` `names_by_type_id`, `discriminators_by_type_id` | action registry | Yes | 1 — registry itself only sees host-linked `inventory` entries (§0); plugin actions are structurally invisible to it regardless of key type, this needs `PortableTypeId` *and* a registration path plugins can actually reach |
| `app.rs:676` `globals_by_type`, `1786-1905` `App::global::<G>()` family | app-level globals | Likely — plugins can define and set their own `Global` types | 2 |
| `app.rs:1100` `Context::emit`, `app.rs:1518` `apply_emit_effect`, `context.rs:392,778` | event type for `cx.emit`/`cx.subscribe` | **Likely already broken, unconfirmed** — `PanelEvent` (`ui::dock::panel`) is emitted by plugin panels (`plugins/vendor/code_editor/.../mod.rs:...impl EventEmitter<PanelEvent> for ScriptEditor`) but `ui` is *also* separately compiled per plugin, so `TypeId::of::<PanelEvent>()` computed in the plugin may not match a host listener's — same mechanism as the confirmed bugs, not yet exercised by a stack trace | 2 — verify first, high suspicion |
| `elements/div.rs:296,338,358` drag/drop payload type, `473,542` | `on_drag`/`on_drop` type matching | Yes — `tab_panel/render.rs` already builds `DragPanel` for cross-panel tab dragging; a plugin panel participating in drag-drop hits this directly | 2 |
| `keymap.rs:21` `binding_indices_by_action_id` | keymap → action lookup | Yes, if a plugin ships default keybindings for its own actions | 2 |
| `app.rs:2352,2361` `remove_asset`/`fetch_asset` cache key | asset cache | Only if a plugin defines its own `Asset` type | 3 — cache-correctness only (miss/duplicate load), not a crash |
| `inspector.rs:227,315,594` | devtools element-state | Only via `feature = "inspector"` | 3 — dev-only |
| `window.rs:8485,8607,8619` `with_element_state` | per-element persistent state | Yes — any element, including plugin-authored custom elements | 2 |

**Tier 1** = confirmed crash mechanism, needs migration first. **Tier 2** =
identical mechanism, not yet observed to crash, migrate before it does.
**Tier 3** = correctness/leak risk without a crash, migrate opportunistically.

---

## 4. Audit: every `thread_local!` in `gpui-ce` today

| Site | Stores | Plugin-reachable? | Verdict |
|---|---|---|---|
| `window.rs:928` `CURRENT_ELEMENT_ARENA` | current draw's arena pointer | **Yes — confirmed crash** | Migrate to `SharedTlsSlot` (§2.2) |
| `platform/cross/platform.rs:53` `ACTIVE_CONTEXT` | raw event-loop/`AppState` pointers | No — platform/windowing internals, plugins never touch this layer | Leave as-is |
| `platform/cross/platform.rs:92` `IDLE_SCOPE` | profiling span for OS event wait | No — same layer | Leave as-is |
| `flamegraph.rs:988` `SPAN_STACK`/`THREAD_RECORDER` | profiler span stack | Possibly, if plugin code calls `wgpui_scope!`/`wgpui_scope_dyn!` in its own render paths | Low priority — fragments profiling data across host/plugin, doesn't crash; migrate if profiling plugin code becomes a real workflow |
| `flamegraph.rs:1142` `FRAME_COUNTERS` | draw-call/atlas counters | No — only touched from renderer-internal code (`WgpuRenderer::draw`, `WgpuAtlas`), not the generic `Render`/`Panel` surface plugins implement | Leave as-is |
| `flamegraph_ui_capture.rs:530` `UI_TREE_RECORDER` | devtools UI-tree capture | Needs investigation — capture may hook per-element paint, which does cross into plugin-compiled code | Investigate before deciding |
| `executor.rs:779` (local `fn thread_id()`) | cached current-thread ID | Yes, but benign — has a graceful `unwrap_or_else` fallback to `thread::current().id()`; a duplicated copy just caches independently, doesn't produce a wrong value | No fix needed |

---

## 5. What gets deleted once this lands

- `entity_map.rs:158-176,279-304` `LeaseUnchecked`/`lease_unchecked`/
  `end_lease_unchecked` and its whole "avoids TypeId checks" doc comment —
  once `AnyEntity::downcast` checks `PortableTypeId`, the *safe* path works
  across the DLL boundary and this unsafe escape hatch has no remaining
  purpose. Check all call sites (`window.rs:2543,2559`) migrate back to the
  checked `Lease`/`read` first.
- `window.rs:1267-1294` `type_name_hash`, `action_name_hash`,
  `action_instance_discriminator` — all three collapse into
  `PortableTypeId::of::<T>()`.
- `crates/ui/wgpui-component/crates/ui/src/dock/panel.rs:255,259`'s
  `ElementArenaScope::enter` calls added today — redundant once §2.2 lands
  (see the note at the end of §2.2: removing these should be the acceptance
  test for the migration, not something done optimistically before it).

---

## 6. Non-goals — considered and rejected

### 6.1 Convert `gpui-ce` to a real shared dylib

The obvious "just make it actually one library" answer. Rejected because a
Rust `dylib` only contains code for the monomorphizations it was built with.
`gpui-ce` cannot ship a `dylib` containing `Entity<ScriptEditor>`, because
`ScriptEditor` doesn't exist until the plugin, built independently and
later, defines it — there is no way to hand a dynamically-loaded downstream
consumer a fresh generic instantiation from an already-compiled `.dll`. This
isn't a build-configuration inconvenience, it's a hard constraint of how
Rust dylibs work, and it's the actual reason the codebase ended up with
"every plugin recompiles its own copy" as the status quo in the first place.

A *non-generic* shim crate built as a real dylib was considered as a
narrower version of this idea (just for the ambient arena pointer) and is
superseded by §2.2 — the OS-TLS slot gets the same "genuinely one instance"
property without a new crate, new `crate-type`, or any build/distribution
changes at all.

### 6.2 Thread `cx`/`window` through `IntoElement::into_any_element`

Would let call sites reach `App`-stored state directly instead of needing
ambient storage at all, fixing §2.2's motivating case at the root. Rejected
as disproportionate: `IntoElement` is implemented by essentially every
widget in `gpui-ce`, `ui`, `wgpui-base`, and every plugin — hundreds of call
sites, all plugin-facing API, for a problem §2.2 already solves without
touching any of them.

---

## 7. Open questions

- **wasm target.** `platform/cross/platform.rs` and others are
  `#[cfg(not(target_family = "wasm"))]`-gated in places, implying a wasm
  build may exist or be planned. wasm has no `dlopen`-style plugin loading,
  so the multi-copy problem this document addresses doesn't apply there —
  `SharedTlsSlot` should probably fall back to a plain `thread_local!` under
  `cfg(target_family = "wasm")` rather than trying to shim OS TLS that
  doesn't exist on that target. Confirm whether a wasm build is real before
  finalizing the `cfg` matrix.
- **Slot lifecycle.** `TlsAlloc`/`pthread_key_create` slots should be freed
  on real process shutdown (`App::shutdown`, `app.rs:867`) for cleanliness;
  low-stakes for a desktop app about to exit, but should be done correctly
  rather than assumed away.
- **Test isolation.** `src/app/test_context.rs` runs multiple `TestAppContext`
  instances, sometimes concurrently interleaved on one thread (the existing
  comment at `window.rs:930` this document already leans on). Confirm the
  `SharedTlsSlot`-per-`App` design (one OS slot allocated per `App::new()`,
  not one global slot for the whole process) preserves that isolation before
  migrating — this should fall out naturally since each `App` gets its own
  slot index, but needs a test that actually exercises two interleaved test
  `App`s to confirm.
- **`ActionRegistry` plugin visibility**, flagged in §3's Tier 1 row: even
  with `PortableTypeId`, the registry itself only ever sees actions
  `inventory`-scanned from the host at startup. Fixing the *key* type doesn't
  give the registry a way to learn about `plugins/vendor/code_editor`'s
  `SaveCurrentFile`/`CloseCurrentFile`. That needs a separate, explicit
  registration step — e.g. a `register_plugin_actions(&[MacroActionData])`
  call the plugin loader invokes once per plugin at load time, using data the
  plugin exports (the same `inventory`-collected list, just handed across the
  FFI boundary explicitly instead of relying on link-time collection to see
  across binaries). Out of scope for this document but a real follow-on;
  today's fix (`action_instance_discriminator`, §0) works without it because
  it never needs the registry to know the plugin's action *names* — only its
  own `.name()` — but `available_actions()`/menu-building/keymap-driven
  dispatch for plugin actions will still need this.

---

## 8. Migration plan

Phased so every step leaves the tree building and passing tests; no
big-bang PR.

1. **Land `PortableTypeId` and `SharedTlsSlot` in isolation** (§2.1, §2.2),
   with unit tests: `PortableTypeId::of::<T>()` stability across a
   `cfg(test)` "simulated second compilation" isn't directly testable in one
   binary, so test it via property (same `T` → same value, different `T` →
   different value, matches `type_name`) and rely on the existing
   confirmed-fixed bugs as the real cross-DLL validation. `SharedTlsSlot`:
   test alloc/get/set round-trip and that two `SharedTlsSlot`s from separate
   `alloc()` calls don't collide.
2. **Migrate Tier 1** (§3): entity downcast, full action-dispatch chain
   including `global_action_listeners`, `ActionRegistry`'s own maps. Delete
   `LeaseUnchecked` and the three superseded hash helpers (§5) once their
   call sites are confirmed migrated.
3. **Migrate the element arena to `SharedTlsSlot`** (§2.2). Acceptance test:
   revert the `panel.rs` patch from today and confirm the script editor
   still opens cleanly.
4. **Investigate and migrate Tier 2** (§3): globals, event emission
   (confirm the `PanelEvent` suspicion first — write a test that subscribes
   from host code to an event emitted by plugin-compiled code and see if it
   currently fires), drag/drop, keymap, element state.
5. **Tier 3 and the `flamegraph_ui_capture` thread-local**, opportunistically.
6. **`ActionRegistry` plugin-visibility follow-on** (§7), as a separate
   document/proposal once this lands — it's a different problem (registry
   completeness, not type identity) that this work makes tractable but
   doesn't itself solve.
