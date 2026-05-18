# `memory/` vs `agent_memory/` — Why Both Exist

This is intentional separation, not duplication. The split mirrors the
**hexagonal / ports-and-adapters** pattern.

## Layers

```
                          ┌──────────────────────────────┐
                          │  agent_memory/   (engine)    │
                          │  ─────────────────────────   │
                          │  MemoryManager + 6 stores    │
                          │  + UserModelEngine           │
                          │  + ConsolidationWorker       │
                          │  + ReflectionEngine          │
                          │  + FactExtractor             │
                          │  + 311 unit tests            │
                          └──────────────────────────────┘
                                        ▲
                                        │  imports
                                        │  (one-way only)
                                        │
   ┌──────────────────────┐      ┌────────────────────┐
   │ runtime/, webui/,    │ ───▶ │  memory/  (port)    │
   │ integrations/, ...   │      │  ─────────────────  │
   │ (callers)            │      │  MemoryAdapter      │
   │                      │      │  + RecallResult     │
   │                      │      │  + operator scope   │
   │                      │      └────────────────────┘
   └──────────────────────┘
```

## Why not one module?

**`agent_memory/` is the engine.** It owns the actual data structures,
SQLite/FTS5 stores, embedding indices, and the algorithms for short/long-term
memory, fact extraction, user modelling, reflection. It has its own test
suite (311 tests). It can evolve internals freely without breaking callers.

**`memory/` is the port.** It exposes the stable, async-first surface that
the rest of the codebase actually calls. Today that's `MemoryAdapter`,
`RecallResult`, and the per-operator scoping helpers
(`set_current_operator`, `get_current_operator`).

This means:

| Concern | Lives in |
|---------|----------|
| Async API + scoping | `memory/` |
| Sync internal stores + algorithms | `agent_memory/` |
| Caller code (`webui`, `runtime`, etc.) | Imports from `memory/` only |
| Tests for engine internals | `agent_memory/tests/` |
| Tests for adapter contract | `memory/tests/` (or absent — adapter is thin) |

## Rules

1. **Callers may NOT import from `agent_memory.*` directly.** Always import
   from `memory`. This keeps the engine swappable. (See audit
   `scripts/audit_directive_parsing.py` for the pattern; an
   `audit_memory_layer.py` could enforce this if violations creep in.)
2. **`memory/` may import from `agent_memory/`.** It's the adapter.
3. **`agent_memory/` may NOT import from `memory/`.** No backwards dependency.
4. **Both packages publish `__all__`.** Anything not in `__all__` is private.

## When to add code where

- **New algorithm or store** → `agent_memory/` (engine layer)
- **New caller-facing helper** (async wrapper, scoping context manager, etc.)
  → `memory/` (port layer)
- **Refactoring the engine internals** → only `agent_memory/`. Callers
  should see no diff.
- **Changing the public adapter API** → both, with the change visible in
  `memory/__init__.py.__all__` for callers to grep.

## Why not collapse them today?

It's tempting because `memory/` is small. But:
- The engine has 311 tests we don't want to mix with adapter tests
- Two callers (`runtime`, `webui`) sometimes need to bypass the operator
  scope (debug endpoints). Keeping the bypass localized in `memory/adapter.py`
  is cleaner than putting it in the engine.
- This split was an explicit refactoring decision after the v5 → v6
  migration; reverting it would walk back a working design.

If you ever want to inline the adapter, the right move is to publish
`agent_memory.facade.Adapter` and have `memory/__init__.py` become a
one-line re-export. The shape is already 80% there.
