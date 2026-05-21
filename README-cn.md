# NetOpYuAgent — Framework Design Document

> *A pluggable, config-driven agent framework for IT operations.*
> *Version: 6.x (post-refactor)*

---

## 1. Design Philosophy

### 1.1 Three guiding principles

The framework was built around three non-negotiable principles:

1. **Module independence** — every functional module (memory, hitl, skill, tool,
   retrieval, schema, evaluation) must be usable in isolation, with no
   imports from any other functional module.
2. **Config-driven composition** — cross-module wiring lives in
   `config.py` / `config.yaml`. No feature is hard-wired; everything has
   an enable flag and tunable knobs.
3. **Verifiable in isolation** — every module ships with a duck-typed
   integration surface (Protocols, simple dicts) so it can be exercised
   with fake collaborators in unit tests, without booting the full app.

### 1.2 Context-layering model

The agent's prompt is composed of four nested layers, in this order:

```
┌────────────────────────────────────────────────────────────────┐
│  System layer        — role, safety rules, output format        │
│                        (fixed across all turns)                 │
├────────────────────────────────────────────────────────────────┤
│  Task layer          — current user query, working set IDs      │
├────────────────────────────────────────────────────────────────┤
│  Memory layer        — recall results: facts, user_profile,     │
│                        confirmed_facts, recent-turn summaries   │
├────────────────────────────────────────────────────────────────┤
│  Capability layer    — top-K skills + top-K tools + meta-tools  │
│                        (selected by retrieval against query)    │
└────────────────────────────────────────────────────────────────┘
```

Each layer is **independently configurable** for size budget, content
shape, and presence (some layers can be empty in stripped-down deployments).

### 1.3 Memory architecture

Memory is organised into four tiers, each with its own backing store,
TTL, and access pattern:

| Tier         | Storage             | Lifespan   | Retrieval         | Purpose                                  |
|--------------|---------------------|------------|-------------------|------------------------------------------|
| short-term   | per-session dict    | session    | direct            | Recent turns; tool results with ref_ids  |
| mid-term     | SQLite + FTS5 + TFIDF | days–weeks | hybrid search    | Extracted facts (LLM-driven)             |
| long-term    | SQLite              | months+    | semantic recall   | Cross-session knowledge                  |
| user_profile | SQLite              | persistent | direct + recall   | User preferences and traits              |

Tool results that are too large to fit in context are kept in the
short-term store with a `[STORED:tool:ref_id]` reference inserted into
the LLM's view. The LLM reads them in pages via `[TOOL:read_stored_result]`.

### 1.4 Skill / Tool architecture

- **Tools** are atomic, callable operations with a JSON-Schema-described
  argument shape. Discovered from three sources: local Python registries,
  MCP servers, OpenAPI specs.
- **Skills** are recipes — markdown procedures the LLM consults when the
  task pattern matches. Skills don't execute; they advise.
- Both are **dynamically injected via retrieval** based on the query;
  not every tool / skill ships in every prompt.

---

## 2. Module Inventory

### 2.1 Independent functional modules

Each row below has **zero imports** from other functional modules (verified
by static analysis).

| Module          | Purpose                                       | Verified independent? |
|-----------------|-----------------------------------------------|-----------------------|
| `agent_memory/` | Multi-tier memory core (fact store, recall)   | ✅ — no cross-module imports |
| `memory/`       | Public adapter wrapping `agent_memory`        | ✅ — only depends on `agent_memory` (proper layering) |
| `hitl_core/`    | Human-in-the-loop interrupt + decision system | ✅ |
| `skills/`       | Skill catalog, evolver, journal, loader       | ✅ — *(post-refactor: tools/ no longer reaches into skills/)* |
| `tools/`        | Tool registries (mock + pragmatic) + loader   | ✅ — *(post-refactor)* |
| `retrieval/`    | Pluggable retriever framework (BM25, embed, hybrid, cache, LLM-judge) | ✅ |
| `schema/`       | Unified arg-schema framework (JSON-Schema subset) | ✅ |
| `evaluation/`   | Golden-set + retrieval benchmark harness      | ✅ |
| `registry/`     | Agent identity + capability registry          | ✅ |
| `runtime/`      | Core agent loop, stop policies, context budget | Imports from skill/memory/tool **only via Protocol** or `services` dict — no concrete coupling |

### 2.2 Cross-module bridges (`integrations/`)

`integrations/` is intentionally NOT a functional module — it contains
glue code that bridges two functional modules, organised by purpose:

```
integrations/
├── clients/        — outbound clients for external protocols
│   ├── llm_engine.py        Ollama/OpenAI/Anthropic LLM clients
│   ├── mcp_client.py        MCP protocol client
│   ├── openapi_client.py    OpenAPI 3 spec parser + invoker
│   └── embedder.py          Embedding-model clients
│
├── adapters/       — optional cross-module bridges (config-gated)
│   ├── memory_facts_adapter.py     SkillJournal → MemoryFacts
│   ├── fact_conflict_detector.py   conflict-aware MemoryFact writes
│   └── hitl_executor.py            A2A protocol ↔ HITL gate
│
└── router/         — unified tool dispatch
    └── tool_router.py        ToolRouter (local + MCP + OpenAPI dispatch)
```

**Backwards compatibility:** all old import paths
(`from integrations import ToolRouter`) still work via
`integrations/__init__.py` re-exports. New code should prefer the
explicit sub-package path (`from integrations.router import ToolRouter`).

### 2.3 Supporting

| Path             | Purpose                                                 |
|------------------|---------------------------------------------------------|
| `webui/`         | FastAPI backend + HTML dashboard (Skills, HITL, Journal) |
| `a2a/`           | Agent-to-agent protocol primitives                       |
| `task/`          | Task graph primitives (inter-task / intra-task)          |
| `data/`          | Runtime data (skills, golden_set, logs, embeddings)      |
| `examples/`      | Documentation + sample uploadable tools                  |
| `tests/`         | Test suite                                               |

---

## 3. Configuration Model

All wiring goes through `config.py` (dataclasses) + `config.yaml` (overrides).

### 3.1 Top-level structure

```python
@dataclass
class AppConfig:
    mode:                str                     # "mock" | "pragmatic"
    memory:              MemoryConfig
    hitl:                HitlConfig
    tools:               ToolsConfig
    retrieval:           RetrievalConfig         # algorithm + thresholds
    skill_orchestration: SkillOrchestrationConfig  # selection, ambiguity HITL, journal
    cross_module:        CrossModuleConfig       # ← inter-module联动
    context_budget:      ContextBudgetConfig     # legacy | priority strategy
    evaluation:          EvaluationConfig        # golden-set runner
    streaming, truncation, hermes, post_verify, session_store, concurrency, ...
```

### 3.2 Cross-module activation (the联动 namespace)

Following the **"联动属于跨功能模块配置项"** principle, all module-to-module
wiring lives under `cfg.cross_module`:

```yaml
cross_module:
  journal_to_facts:                    # SkillJournal → MemoryFacts
    enabled:           false           # default OFF — module pairs are independent
    interval_s:        600
    dormant_threshold: 0.6
    success_threshold: 0.9
    fact_ttl_days:     14.0

  fact_conflict_detection:             # conflict-aware MemoryFact writes
    enabled:                false
    similarity_threshold:   0.70
    equivalence_threshold:  0.85
    llm_reconcile_enabled:  false
```

**Invariant:** any cross_module feature MUST be safe to disable with no
functional regression in other modules. Tested at startup.

### 3.3 Pluggable algorithm selection

Algorithms are functional-module configuration (not cross-module):

```yaml
retrieval:
  backend: "hybrid"     # bm25 | embedding | hybrid | keyword | llm_judge
  hybrid:
    bm25_weight:  0.5
    embed_weight: 0.5
    fusion:       "weighted_sum"      # | "rrf"
  cache:
    enabled: true
    ttl_seconds: 600

memory:
  embedding_backend: "ollama"          # tfidf | ollama | openai | stub

context_budget:
  strategy: "legacy"   # "legacy" (current behaviour) | "priority" (P0/P1/P2/P3)
```

### 3.4 Subagent / capability config

Following **"subagent 属于框架能力配置项"**:

```yaml
# Reserved for future implementation
skill_orchestration:
  subagent:
    enabled: false                    # not yet implemented
    triggered_by: ["high_complexity"] # when to delegate to a sub-agent
    max_concurrent: 3
    timeout_s: 120
```

### 3.5 Environment variable double-path

Every config option supports an env var override following the pattern
`<MODULE>_<OPTION>`. Examples:

```bash
RETRIEVAL_BACKEND=llm_judge
SKILL_HITL_ON_AMBIGUITY=true
XM_JOURNAL_TO_FACTS=true
CTX_BUDGET_STRATEGY=priority
```

This lets operators tune deployments without touching YAML.

---

## 4. Module Contracts

### 4.1 Memory module

**Public surface (`memory.MemoryAdapter`):**
```python
async def add_fact(session_id, user_id, fact_text, *, fact_type, confidence,
                   metadata, ttl_days) -> str            # returns fact_id
async def find_similar_facts(user_id, query_text, *, session_id, fact_type,
                             top_k) -> list[dict]
async def update_fact_confidence(fact_id, new_confidence, *, reason) -> bool
async def after_turn(...) -> None                       # runtime hook
async def recall(...) -> list[MemoryFact]
```

External modules (skill journal adapter, fact conflict detector) **only**
use these public methods. Internal `_mgr` and `_store` are off-limits.

### 4.2 Skill module

**Public surface:**
```python
SkillLoader(mode).skill_definitions() -> dict[str, dict]
SkillCatalogService.register_all(defs)
SkillCatalogService.attach_retriever(retriever)         # plug retrieval algorithm
SkillCatalogService.select_skills_for_query(query, top_k) -> SkillSelectionResult
SkillJournal.record_*()                                  # observability
SkillEvolver.apply_feedback(skill_id, feedback, success) # self-improvement
```

The catalog uses **any retriever** that implements
`Retriever.retrieve(query, top_k) -> RetrievalResult`. Algorithm
selection is config-driven.

### 4.3 Tool module

**Public surface:**
```python
ToolLoader(mode).build_metadata() -> dict[str, dict]   # tool metadata
ToolLoader(mode).get_callable(name) -> Callable
ToolRouter.dispatch(tool_name, args) -> str
```

ToolRouter accepts callables from **three sources** through a uniform
metadata format: local registry, MCP server, OpenAPI operation. Schema
validation (`schema/`) runs as opt-in middleware before dispatch.

### 4.4 HITL module

**Public surface (`hitl_core`):**
```python
HitlGate.create_interrupt(interrupt_kind, ...) -> InterruptId
HitlGate.deliver_decision(interrupt_id, decision)
HitlGate.await_decision(interrupt_id) -> Decision
```

Currently three interrupt kinds: `tool_call`, `user_choice`,
`clarification`. New kinds are added by registering a `Resumer` —
no changes to the core.

### 4.5 Retrieval module

```python
Retriever.index(items)                                  # indexes a corpus
Retriever.retrieve(query, top_k, ...) -> RetrievalResult
Retriever.corpus -> list[dict]                          # introspection
```

Five built-in retrievers:

| Retriever          | Strength                          | When to use                  |
|--------------------|-----------------------------------|------------------------------|
| `BM25Retriever`    | Lexical, CJK-friendly tokeniser   | Default lightweight          |
| `EmbeddingRetriever` | Cross-lingual semantic similarity | When embeddings available    |
| `HybridRetriever`  | BM25 + embedding fusion           | **Recommended production**   |
| `CachedRetriever`  | LRU+TTL wrapper for any retriever | Always (zero-cost)           |
| `LLMJudgeRetriever`| Two-stage rerank with LLM         | Highest quality, higher latency |

---

## 5. Runtime Lifecycle

```
┌───────────────────────────────────────────────────────────────────┐
│  Startup (main.py)                                                │
│  1. Load config (YAML + env overrides)                            │
│  2. Init each module independently:                               │
│       - SkillLoader / ToolLoader build metadata                   │
│       - MemoryAdapter binds backends                              │
│       - HitlGate creates store                                    │
│       - Retrievers index corpora (async)                          │
│       - SchemaRegistry imports from MCP/OpenAPI/dict              │
│  3. Wire optional cross_module adapters (if enabled in cfg):      │
│       - JournalToFactsAdapter (background task)                   │
│       - FactConflictDetector (services injection)                 │
│       - SkillJournalConsumer (background task)                    │
│  4. Hook up runtime loop via patch_runtime_loop()                 │
│  5. Mount HTTP routes (webui)                                     │
└───────────────────────────────────────────────────────────────────┘
                                ↓
┌───────────────────────────────────────────────────────────────────┐
│  Per-query (runtime/loop.py)                                      │
│  1. Build state.skill_journal (observability)                     │
│  2. Retrieve skills + tools → inject into prompt                  │
│  3. Recall memory facts → inject into prompt                      │
│  4. While not done:                                                │
│     a. LLM completes → parse [TOOL:...] / [SKILL_LOAD:...]        │
│     b. If tool call: ToolRouter dispatches → may pause for HITL   │
│     c. Update working_set, confirmed_facts, journal              │
│     d. Decide stop policy: complete / step / nudge                │
│  5. Finalise journal → persist + push to evolver feedback queue   │
└───────────────────────────────────────────────────────────────────┘
```

---

## 6. How the Framework Stays Pluggable

### 6.1 Swap a memory backend

Change `cfg.memory.embedding_backend` from `tfidf` to `ollama`. No code
changes. The `MemoryAdapter` rebinds; everything downstream is unchanged.

### 6.2 Swap a retrieval algorithm

Change `cfg.retrieval.backend` to any of `bm25 | embedding | hybrid |
keyword | llm_judge`. `build_skill_retriever()` constructs the chosen
class; the catalog uses whatever it gets.

### 6.3 Enable cross-module联动

```yaml
cross_module:
  journal_to_facts:
    enabled: true
```

The startup wiring detects the toggle, instantiates the adapter, attaches
it to existing modules through their public APIs. **No functional module
is modified.**

### 6.4 Add a new tool source

Implement `ToolRouter.register_*()`. Tool routing pipeline is shared:
metadata → schema validation → dispatch → result formatting.

### 6.5 Add a new skill mode

Drop `skills/mymode/registry.py` with a `SKILLS` dict. Update
`SkillLoader.skill_definitions()` (one if-branch). Done.

---

## 7. Project Directory Layout (post-refactor)

```
NetOpYuAgent/
├── main.py                            # Entrypoint
├── config.py                          # Config dataclasses + loader
├── config.yaml                        # Operator-tunable settings
│
├── agent_memory/                      # ★ Independent: memory core
│   ├── stores/                        #   {short,mid,long}_term_store, user_model
│   ├── retrieval/                     #   recall orchestrator (memory-internal)
│   ├── consolidator.py                #   summary/compression
│   └── memory_manager.py
│
├── memory/                            # ★ Public adapter for agent_memory
│   └── adapter.py
│
├── hitl_core/                         # ★ Independent: HITL gate
│   ├── pipeline.py                    #   _DecisionWaiter, BoundedSessionStore
│   ├── store.py                       #   persistence (memory/sqlite)
│   ├── router.py                      #   decision delivery
│   ├── resumers/                      #   tool_call, user_choice, clarification
│   └── transport/
│
├── skills/                            # ★ Independent: skill catalog
│   ├── loader.py                      #   SkillLoader (mode-aware)
│   ├── catalog.py                     #   SkillCatalogService
│   ├── evolver.py                     #   SkillEvolver (self-improvement)
│   ├── journal_consumer.py            #   Journal → Evolver feedback
│   ├── builtin/registry.py
│   ├── mock/registry.py
│   └── pragmatic/registry.py
│
├── tools/                             # ★ Independent: tool registries
│   ├── loader.py                      #   ToolLoader (mode-aware)
│   ├── mock_tools.py
│   ├── pragmatic_tools.py
│   ├── builtin/registry.py
│   ├── mock/registry.py
│   └── pragmatic/registry.py
│
├── retrieval/                         # ★ Independent: pluggable retrieval
│   ├── base.py                        #   Retriever interface, Match, RetrievalResult
│   ├── bm25.py
│   ├── embedding.py
│   ├── hybrid.py
│   ├── cache.py
│   ├── llm_judge.py
│   ├── keyword.py
│   ├── factory.py                     #   build_*_retriever
│   └── meta_tool.py                   #   list_tools / list_skills / tool_details
│
├── schema/                            # ★ Independent: unified arg-schema
│   ├── types.py                       #   ArgSchema, ArgField, FieldType
│   ├── validator.py                   #   validate_and_coerce
│   ├── importers.py                   #   from_{dict,mcp,openapi}
│   ├── prompt.py                      #   render_args_for_prompt
│   └── registry.py                    #   process-wide registry
│
├── evaluation/                        # ★ Independent: eval harness
│   ├── types.py                       #   EvalCase, BenchReport
│   ├── golden_set.py                  #   JSONL load/save/validate
│   ├── retrieval_bench.py             #   RetrievalBench
│   ├── reporters.py                   #   text/JSONL output
│   └── cli.py                         #   `python -m evaluation.cli ...`
│
├── runtime/                           # Agent loop
│   ├── loop.py                        #   AgentRuntimeLoop.stream()
│   ├── stop_policy.py                 #   StopOutcome, FactsLedger
│   ├── context_budget.py              #   Legacy length-based budget
│   ├── context_budget_v2.py           #   ★ Priority-based budget (new)
│   ├── policy_engine.py               #   classifier policies
│   └── skill_journal.py               #   SkillJournal (observability)
│
├── integrations/                      # Cross-module glue (organised by purpose)
│   ├── clients/                       #   ─ outbound external protocols
│   │   ├── llm_engine.py              #     OllamaEngine, OpenAIEngine, …
│   │   ├── mcp_client.py
│   │   ├── openapi_client.py
│   │   └── embedder.py
│   ├── adapters/                      #   ─ optional cross-module bridges
│   │   ├── memory_facts_adapter.py    #     Journal → MemoryFacts
│   │   ├── fact_conflict_detector.py  #     fact write reconciliation
│   │   └── hitl_executor.py           #     A2A protocol ↔ HITL
│   └── router/                        #   ─ unified tool dispatch
│       └── tool_router.py
│
├── registry/                          # Agent identity + peer registry
│
├── webui/                             # FastAPI dashboard
│   ├── backend.py                     #   routes
│   ├── index.html                     #   UI (tabs: Results, Flow, HITL,
│   │                                    #     Memory, Stats, Journal)
│   └── static/
│
├── a2a/                               # Agent-to-agent protocol primitives
├── task/                              # Task graph primitives
│
├── data/                              # Runtime data
│   ├── skills/                        #   evolver-persisted skill markdown
│   ├── embeddings/                    #   on-disk embedding cache
│   ├── memory/                        #   SQLite DBs
│   └── golden_set.jsonl               #   eval ground truth
│
├── examples/                          # Documentation + sample uploads
│   └── uploads/                       #   user-uploadable tool/skill examples
│
└── tests/                             # Test suite
```

---

## 8. Where the Framework Stands Today

### 8.1 Implemented (cf. design intent)

| Design intent                                  | Status                  |
|------------------------------------------------|-------------------------|
| Independent functional modules                 | ✅ all 10 modules clean (verified by import audit) |
| Config-driven module activation                | ✅ AppConfig + YAML + env vars |
| Pluggable retrieval                            | ✅ 5 backends + cache wrapper |
| Pluggable memory embedding                     | ✅ tfidf / ollama / openai |
| Pluggable HITL transport                       | ✅ SSE / file / memory  |
| Cross-module联动 namespace                     | ✅ `cfg.cross_module.*` |
| Context layering (system/task/memory/capability) | ✅                     |
| Multi-tier memory (short/mid/long/profile)     | ✅                     |
| Tool result external storage + paged read      | ✅                     |
| Skill journal + evolver feedback loop          | ✅                     |
| Schema framework (MCP/OpenAPI/dict unified)    | ✅                     |
| Eval harness with golden set                   | ✅                     |
| Priority-based context budget                  | ✅ (opt-in, `cfg.context_budget.strategy=priority`) |

### 8.2 Reserved / not yet implemented

| Capability                                  | Reason for deferral             |
|---------------------------------------------|---------------------------------|
| Skill as sub-agent (LangGraph subgraph)     | Requires LLM-routing redesign; reserve `cfg.skill_orchestration.subagent.*` namespace |
| LLM-managed memory (MemGPT-style)           | Higher cost, lower deterministic behaviour; wait for journal data to justify |
| Per-tool circuit breakers                   | Implemented partially in ToolMeta — needs UI integration |
| Distributed multi-agent (A2A)               | A2A primitives exist; orchestration layer pending |

### 8.3 What was refactored in this iteration

1. **SkillLoader extracted from ToolLoader**
   - Previously: `tools/loader.py` imported from `skills/*/registry.py`,
     violating module independence.
   - Now: `skills/loader.py:SkillLoader` owns skill loading. `ToolLoader.skill_definitions()`
     remains as a deprecated shim that emits a `DeprecationWarning`.

2. **`integrations/` reorganised by purpose**
   - Previously: `integrations/` was a flat folder mixing protocol clients,
     adapters, and routing.
   - Now: `integrations/{clients,adapters,router}/` sub-packages by purpose.
   - Backward compatible: `from integrations import ToolRouter` still works.

3. **`mock_file/` cruft removed**
   - Moved to `examples/uploads/` since it contained user-uploadable sample
     tools, not framework code.

---

## 9. Test Coverage Summary

| Module                              | Independent unit verification | Full integration |
|-------------------------------------|-------------------------------|------------------|
| `agent_memory/`                     | ✅ `agent_memory/tests/`      | ✅ via webui Q&A |
| `hitl_core/`                        | ✅ existing tests             | ✅ via webui HITL approval flow |
| `skills.SkillCatalogService`        | ✅ scoring + ambiguity test   | ✅ via runtime  |
| `skills.SkillJournal`               | ✅ this iteration             | ✅ via runtime  |
| `skills.SkillJournalConsumer`       | ✅ this iteration             | ⚠ background only |
| `tools.ToolLoader` / `ToolRouter`   | ✅                            | ✅              |
| `retrieval/` (all 5 backends)       | ✅                            | ✅              |
| `schema/` validator + importers     | ✅                            | ✅              |
| `evaluation/`                       | ✅ this iteration             | ✅ via CLI       |
| `integrations.adapters.memory_facts`| ✅ this iteration (fake stores)| ⚠ opt-in only  |
| `integrations.adapters.fact_conflict_detector` | ✅ this iteration  | ⚠ opt-in only  |
| `runtime.TokenBudget` (priority)    | ✅ this iteration             | ⚠ opt-in only  |

---

## 10. Operations Cheatsheet

### 10.1 Enable cross-module learning

```yaml
# Have journal observations promote to long-lived memory facts
cross_module:
  journal_to_facts:
    enabled: true
```

### 10.2 Enable fact conflict detection

```yaml
cross_module:
  fact_conflict_detection:
    enabled: true
    llm_reconcile_enabled: false   # cheap heuristic only
```

### 10.3 Run a retrieval bench

```bash
python -m evaluation.cli \
    --golden data/golden_set.jsonl \
    --backend hybrid \
    --kind skill \
    --top-k 5 \
    --fail-below-mrr 0.5
```

### 10.4 Switch context budget strategies

```yaml
context_budget:
  strategy: "priority"
  total_chars: 64000
  section_recent_turns: 25000
  section_retrieved_memory: 12000
```

### 10.5 View skill journal

WebUI → Journal tab. Endpoints:
- `GET /webui/skill_journal/recent`
- `GET /webui/skill_journal/stats`
- `GET /webui/skill_journal/filter?skill_id=X&outcome=Y&ambiguous=true`

---

## 11. Glossary

| Term                  | Definition                                                          |
|-----------------------|---------------------------------------------------------------------|
| **Functional module** | An independent code unit with no cross-module imports (`memory/`, `hitl_core/`, etc.) |
| **Adapter**           | An opt-in bridge connecting two functional modules, lives in `integrations/adapters/` |
| **Skill**             | A markdown procedure consulted by the LLM (not executable code)     |
| **Tool**              | An atomic callable (local fn, MCP server method, or OpenAPI op)     |
| **Meta-tool**         | A tool the LLM uses to discover other tools (`list_tools`, etc.)    |
| **Journal**           | Observability log of skill load + tool call events per session      |
| **Fact**              | Structured statement extracted from conversation, stored in mid-term memory |
| **User profile**      | Long-lived inferred traits + explicit preferences per user          |
| **Retriever**         | Pluggable ranking module (BM25, embedding, hybrid, etc.)             |
| **HITL gate**         | Mechanism that pauses the agent until a human approves an action    |
| **Subagent**          | (Reserved) Independent LLM instance handling a sub-task             |

---

## 12. Multi-Agent (Phase 1 — identity + peer discovery)

Phase 1 establishes the foundation for multiple agent processes to discover each other. The A2A protocol layer (inbound server + outbound dispatcher) has always been in place; what was missing was per-process identity and active peer discovery. Phase 1 closes that gap. **Cross-agent dispatch and cross-agent HITL passthrough are Phase 2 and Phase 3 respectively — not yet wired.**

### 12.1 `cfg.agent` namespace

`AgentIdentityConfig` (in `config.py`) carries per-process identity:

```python
@dataclass
class AgentIdentityConfig:
    agent_id:                str = "default-agent"
    display_name:            str = "IT Ops Agent"
    description:             str = "..."
    capabilities:            list[AgentSkillSpec] = field(default_factory=list)
    peer_urls:               list[str] = field(default_factory=list)
    peer_refresh_interval_s: int = 120
```

`AgentSkillSpec` is one row of the `AgentCard.skills` array — `skill_id`, `description`, `tags`, `examples`. Phase 2 capability matching will rank peers by tag overlap against the query.

### 12.2 Configuration precedence

1. Env vars (`AGENT_ID`, `AGENT_DISPLAY_NAME`, `AGENT_DESCRIPTION`, `AGENT_PEERS`, `AGENT_PEER_REFRESH_S`)
2. `config.yaml` `agent:` section
3. Defaults (reproduce legacy single-agent behaviour)

Defaults are tuned so omitting the `agent:` block entirely keeps existing single-agent deployments working without yaml changes.

### 12.3 What's wired at startup

`main.py:build_services` now:

1. Builds the AgentCard with `get_agent_card(base_url, identity=cfg.agent)`
2. Merges peer URLs from both `cfg.registry.agent_urls` (legacy) and `cfg.agent.peer_urls` (new), deduped + order-preserving
3. Passes the merged list to `create_registry(static_urls=...)` so peer AgentCards are fetched + indexed at boot
4. Stashes the merged list in `services["_peer_urls"]` for the lifespan refresh loop

`main.py:lifespan` adds a background task that re-fetches peer AgentCards every `peer_refresh_interval_s` seconds. Failures are logged but never raised (a momentarily-down peer should not kill the agent).

### 12.4 `/system/peers` endpoint

```bash
GET /webui/system/peers
```

Returns the registered peers excluding self, plus this agent's own identity:

```json
{
  "self": {
    "agent_id": "lan-agent",
    "display_name": "LAN Agent",
    "url": "http://localhost:8000/api/v1/a2a",
    "capabilities": ["lan_diagnose", "lan_config"]
  },
  "peers": [
    {
      "agent_id": "wan-agent",
      "agent_url": "http://localhost:8001/api/v1/a2a",
      "display_name": "WAN Agent",
      "health": "healthy",
      "capabilities": ["wan_diagnose", "bgp_query", "wan_config"]
    }
  ],
  "peer_refresh_interval_s": 120
}
```

Use this as the single source of truth when verifying multi-agent topology before attempting any cross-agent work.

### 12.5 Phase 1 scope boundary

In scope:
- ✅ AgentCard identity-driven (skills come from `cfg.agent.capabilities` when set)
- ✅ Peer discovery via existing `AgentDiscovery.fetch_card`
- ✅ Periodic peer refresh
- ✅ `/system/peers` endpoint
- ✅ 16/16 unit tests in `tests/test_multi_agent_identity.py`

Out of scope (Phase 2+):
- ❌ Routing a query from one agent to another (`A2ATaskDispatcher` exists but is not yet called from `runtime/loop.py`)
- ❌ Capability-based peer matching in PolicyEngine
- ❌ Cross-agent HITL passthrough (`INPUT_REQUIRED` state riding on A2A)
- ❌ Cross-agent confirmed_facts / session context sharing

See `ARCHITECTURE.md §8.3` for the Phase 1 changelog and `§11.3` for the multi-agent roadmap.

---

## 13. Production readiness (Sprint-3-pre, 2026-05)

This release ships four production-readiness foundations. They don't change agent behaviour but make the agent **safer to deploy**. Full design in `ARCHITECTURE.md §8.4`.

### 13.1 HITL approval persistence (sqlite default)

Pending HITL approvals previously lived in an in-memory dict — any restart while an operator was reviewing a destructive-action card lost the producing query's `asyncio.Future` permanently. Now default is sqlite-backed and survives restart.

```yaml
hitl:
  checkpoint:
    backend:     "sqlite"          # memory | sqlite | redis
    sqlite_path: "data/hitl_checkpoints.db"
    redis_url:   ""
```

In-memory + pragmatic mode logs a loud startup warning.

### 13.2 Graceful shutdown drain

`lifespan` registers SIGTERM/SIGINT signal handlers and drains in-flight LLM/tool tasks (max 30s) before exiting. The HITL checkpoint store is flushed before background services stop. `docker stop` / `kubectl rollout` now does the right thing.

Tasks join the drain set by registering in `services["in_flight_tasks"]`; the chat-stream executor in `webui/backend.py` does this automatically.

### 13.3 OpenTelemetry tracing (opt-in)

`runtime/tracing.py` provides a no-op shim — `with start_span("llm.call", ...)` callsites work whether OTel is installed or not. Three high-value spans are wired today:

| Span | Where | Attributes |
|------|-------|------------|
| `agent.query` | `hitl_executor.execute_query` | `agent.session_id`, query chars, facts count |
| `llm.call` | `llm_engine._chat` | `llm.model`, message count, native tools flag |
| `tool.dispatch` | `runtime.loop._dispatch_tool` | tool name, args count, result chars |

```yaml
observability:
  tracing_enabled:  false           # default OFF
  otlp_endpoint:    ""              # "http://collector:4317" or empty (=console)
  sample_ratio:     1.0
```

To enable:

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
OTEL_TRACING_ENABLED=true \
  OTEL_EXPORTER_OTLP_ENDPOINT=http://your-collector:4317 \
  uvicorn main:app --port 8000
```

### 13.4 SkillEvolver A/B safety net

When the agent boots, the compliance bench is wired into `SkillEvolver.set_bench_runner()`. Before any feedback patch is applied:

1. Pick 3-5 compliance cases whose `expected_tool` appears in the new content
2. Bench baseline + candidate via `ToolComplianceBench`
3. If `args_ok` would DROP, patch is rejected and old skill kept; warning logged
4. If equal or better, patch applies normally

When the golden set is missing or LLM engine isn't ready, the safety net silently disables (legacy unchecked path) — best-effort, no boot failure.

### What Sprint-3-pre does NOT fix

Still pending for real Sprint 3 / production rollout:

- ❌ Docker / docker-compose / k8s deployment manifests
- ❌ Auth-required-in-production startup check
- ❌ `/livez` + `/readyz` endpoints that check dependencies
- ❌ Prometheus `/metrics` endpoint (OpenMetrics format)
- ❌ FastAPI / httpx auto-instrumentation (3 manual span sites only)
- ❌ Database migration framework
- ❌ Backup cron + restore runbook
- ❌ CSRF protection on `/hitl/*` POST routes
- ❌ Secrets management (Vault / AWS SM)

Until those land, the agent is safer for **internal / dev / staging** but should not face internet traffic.
