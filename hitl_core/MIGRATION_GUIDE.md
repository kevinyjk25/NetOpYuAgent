# HITL Migration Guide — LangGraph → hitl_core

This guide walks through migrating the project from the legacy LangGraph-
based HITL stack (`hitl/`) to the self-contained `hitl_core/` module.

The migration is fully reversible at every step until **Phase 7C** (final
deletion).

## Phase summary

| Phase | What happens | Reversible? |
|-------|-------------|-------------|
| 1–5 | Build `hitl_core/` + `integrations/hitl_executor.py`. Old code untouched. | yes |
| 6 | `HITL_BACKEND` env var. Default still `langgraph`. | yes |
| 7A (now) | Run with `HITL_BACKEND=core` in staging. Validate every workflow. | yes |
| 7B | Switch production default to `core`. Keep `hitl/` directory. | yes |
| 7C | Delete `hitl/` directory + remove langgraph from `requirements.txt`. | **no** |

## Phase 7A — Staging validation (start here)

**1. Switch the env var:**

```bash
export HITL_BACKEND=core
# Optional: pick a checkpoint backend. Default is in-memory.
export HITL_CHECKPOINT_BACKEND=memory   # or redis / sqlite
# For Redis:
export HITL_REDIS_URL=redis://localhost:6379/0
# For SQLite:
export HITL_SQLITE_PATH=data/hitl_checkpoints.db
# Optional: write audit log to a file
export HITL_AUDIT_LOG_PATH=/var/log/hitl/audit.jsonl

python main.py
```

Logs should show:
```
HITL backend selected: core
HITL checkpoint backend: memory
HITL module ready (backend=core, 0 step(s) wired)
A2A executor (core) constructed with N tool(s)
```

**2. Run the same regression suite you'd run for any HITL change:**

- Single destructive HITL (e.g. "restart svc-payments")
- Nested HITL (e.g. "fix the config" → intent approval → tool approval with EDIT)
- USER_CHOICE multi-mode (e.g. ambiguous "fix it")
- CLARIFICATION multi-mode (e.g. missing device_id)
- Approve, Reject, Edit, Choose flows
- Coreference scenarios (query without device_id after recent tool call)
- Memory writeback after `[HITL APPROVED & COMPLETED]`

**3. Roll back if anything is wrong:**

```bash
unset HITL_BACKEND   # or export HITL_BACKEND=langgraph
```

The old code path is untouched.

## Phase 7B — Production switch

Once 7A passes for at least 1 week:

1. Update production env vars to default `HITL_BACKEND=core`.
2. Don't delete anything yet — keep the langgraph path as a safety hatch.
3. Watch for 1-2 release cycles.

## Phase 7C — Final cleanup (irreversible)

Only do this after 7B has been stable. The full deletion is:

**Files to delete:**
```
hitl/__init__.py
hitl/graph.py                    # 823 lines, the LangGraph state machine
hitl/a2a_integration.py          # 2072 lines, replaced by integrations/hitl_executor.py
hitl/audit.py                    # replaced by hitl_core/audit.py
hitl/decision.py                 # replaced by hitl_core/router.py
hitl/router.py                   # replaced by hitl_core/router.py
hitl/review.py                   # replaced by hitl_core/transport/sse_adapter.py
hitl/schemas.py                  # replaced by hitl_core/schema.py
hitl/triggers.py                 # replaced by hitl_core/triggers.py
agent_memory/examples/langchain_integration.py  # commented-out example only
```

**Code edits in non-deleted files:**

`main.py` — remove the `else:` branch under `if _hitl_backend == "core"`,
keeping only the `core` path. Search for the comment
"Legacy path — hitl/* + LangGraph" and delete that whole block.

`webui/backend.py` — same thing: remove the `else:` branch in
`_submit_hitl_decision` (the legacy `hitl_router._payload_store` lookup) and
the legacy block in `/hitl/pending`.

`task/inter/hitl_bridge.py` — currently imports `hitl.decision` /
`hitl.review` / `hitl.schemas`. Either delete this file (if task-level
HITL is not used) or rewrite using `hitl_core` primitives.

`integrations/llm_engine.py` line ~948 — remove the `import hitl.graph`
inside `patch_hitl_graph()` and remove the function entirely.

`requirements.txt` — comment out or delete:
```
langgraph>=0.2.0
langchain-core>=0.3.0
langchain-openai>=0.1.0
```

**Verify:**
```bash
pip uninstall langgraph langchain-core langchain-openai
grep -rn "from hitl\|import hitl[^_]" --include="*.py" | grep -v hitl_core
# Should return zero matches outside of test fixtures.
HITL_BACKEND=core python main.py
```

## Architecture reference

After full migration:

```
hitl_core/                          ← portable, langchain-free
├── schema.py                       ← Pydantic models
├── store.py                        ← InMemory / Redis / SQLite checkpointers
├── pipeline.py                     ← async pipeline replacing StateGraph
├── router.py                       ← decision validation + dispatch
├── batch.py                        ← batch approval coordinator
├── triggers.py                     ← pluggable trigger engine
├── audit.py                        ← append-only audit log
├── coreference.py                  ← focus-entity inference
└── transport/
    ├── http_adapter.py             ← FastAPI router factory (optional)
    └── sse_adapter.py              ← SSE streaming (optional)

integrations/
└── hitl_executor.py                ← IT-ops business glue (729 lines)

main.py                              ← env-driven backend selection
webui/backend.py                     ← transport layer (handles both backends)
```

## Net code reduction after Phase 7C

```
Removed:
  hitl/graph.py             823
  hitl/a2a_integration.py  2072
  hitl/audit.py             377
  hitl/decision.py          571
  hitl/router.py            431
  hitl/review.py            574
  hitl/schemas.py           282
  hitl/triggers.py          387
  hitl/__init__.py          119
                          -----
                          5636 lines removed

Added:
  hitl_core/ (all 11 files)  4156
  integrations/hitl_executor 729
                            -----
                            4885 lines added

Net:                        -751 lines
Plus: removal of langgraph + langchain-core + langchain-openai
      from requirements.txt (~80 transitive deps).
```

## When NOT to migrate

You can stay on `langgraph` indefinitely if:
- Your team is already deeply familiar with LangGraph and prefers its
  graph-based mental model
- You depend on LangChain ecosystem features beyond what hitl_core provides
- You haven't hit any of the bugs that motivated this refactor (nested
  HITL workarounds, opaque graph state debugging, etc.)

The `HITL_BACKEND=langgraph` path is fully maintained alongside `core`.
