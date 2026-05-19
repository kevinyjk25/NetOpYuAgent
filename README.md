# NetOpYuAgent — IT Ops Multi-Agent Platform

> A pluggable, config-driven multi-agent framework for IT/network operations.
> Built around **module independence**, **HITL safety gates**, and **measurable quality**.

**Languages:** [English](README.md) · [中文](README-cn.md)

---

## What is this

NetOpYuAgent is a production-style AI agent platform for IT operations. It runs locally on Ollama (or any OpenAI-compatible LLM), pairs an autonomous task loop with a human-in-the-loop (HITL) approval gate for destructive operations, and measures itself with a built-in golden-set eval framework.

The platform is **not** a chatbot. It is a runtime that:

- Pulls **relevant tools and skills** per query via BM25+embedding hybrid retrieval (not "send everything to the LLM")
- Pauses on destructive operations (`edit_device_config`, `rollback`, `restart_service`, etc.) and waits for an operator to approve/reject/modify, then resumes
- Maintains **five tiers of memory** (short-term per-session, mid-term facts, long-term knowledge, user profile, skill journal) with semantic recall
- **Learns from every turn**: extracts facts, updates user model, evolves reusable skills, detects fact conflicts
- Ships with **6 CI audits** + a retrieval bench + a tool-compliance bench so quality is measurable, not vibes

---

## Documentation map

This README is the shallow entry point. For depth, the project has three layers of documentation:

| Level | Document | Use when... |
|---|---|---|
| **L0 — Onboarding** | `README.md` (you are here) | First time looking at the project, want to run it |
| **L1 — Architecture** | `ARCHITECTURE.md` | Cross-module change, want the dependency graph + module table |
| **L2 — Module deep-dive** | `<module>/DESIGN.md` | Single-module change, want internal data flow + decision history |

Each functional module ships a `DESIGN.md` with 6 standard sections (responsibility / public API / data flow / design decisions / cross-module deps / change checklist). Current set:
`agent_memory/`, `hitl_core/`, `integrations/`, `retrieval/`, `runtime/`, `skills/`, `tools/`.

---

## Quick Start

### Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally (for real LLM) — or `LLM_BACKEND=mock` for a stub
- ~16 GB RAM for `qwen3.5:27b`, ~6 GB for `qwen2.5:7b`

### Install

```bash
git clone <repo>
cd NetOpYuAgent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run with mock LLM (no Ollama needed)

```bash
LLM_BACKEND=mock uvicorn main:app --port 8000
```

Then open http://localhost:8000 — you'll get the WebUI console.

### Run with Ollama (recommended)

```bash
# Pull the model first (one-time):
ollama pull qwen2.5:7b
ollama pull nomic-embed-text     # embeddings

# Start the agent:
LLM_BACKEND=ollama LLM_MODEL=qwen2.5:7b uvicorn main:app --port 8000
```

For larger / slower model:
```bash
LLM_MODEL=qwen3.5:27b uvicorn main:app --port 8000
```

### Verify it works

```bash
# 1. Health check
curl http://localhost:8000/health
# → {"status":"ok"}

# 2. Try a query via WebUI at http://localhost:8000
#    "list all devices" — should return mock device list, no HITL
#    "restart nginx on web-01" — should trigger HITL approval card
```

---

## Core capabilities

| Capability | Where it lives | Why it matters |
|---|---|---|
| **Dual-path routing** | `runtime/policy_engine.py` | Read-only queries skip LLM evaluation (8000× faster); destructive queries go through HITL |
| **HITL approval gate** | `hitl_core/` + `integrations/adapters/hitl_executor.py` | LangGraph-style `interrupt()` — browser shows approval card, agent resumes on decision |
| **5-tier memory** | `agent_memory/` | Short/mid/long-term + user profile + skill journal; FTS5 + embedding hybrid recall |
| **Memory consolidation** | `MemoryAdapter.set_consolidator` | Every 30 turns per-session, old chunks auto-summarised; long sessions stay fast |
| **Fact conflict detection** | `integrations/adapters/fact_conflict_detector.py` | Before writing a fact, find semantically similar; LLM reconciles contradictions |
| **Skill catalog + evolver** | `skills/` | L1 summaries always present, L2 details loaded on demand; SkillEvolver auto-improves dormant skills |
| **Native tool calls** | `integrations/clients/llm_engine.py` + `schema/ollama_export.py` | (Opt-in) Ollama OpenAI-style `tools` API — structurally eliminates "args filled wrong" failures |
| **Context budget** | `runtime/context_budget.py` | Priority allocation: confirmed_facts > working_set > memory > tool_outputs > env |
| **Stop policy** | `runtime/stop_policy.py` | Six dimensions (max turns / tool calls / token budget / progress / confidence / explicit signal) |
| **MCP + OpenAPI tools** | `integrations/router/tool_router.py` | Plug-in MCP servers and any OpenAPI 3.0 spec |

---

## Architecture (one diagram)

```
External caller (RouterAgent / WebUI / webhook / curl)
       │  A2A JSON-RPC over HTTP-SSE / REST
       ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI (main:app, uvicorn)                 │
│  /api/v1/a2a/*   /hitl/*   /registry/*   /webui/*       │
└───────────────────────────┬─────────────────────────────┘
                            │
            ┌───────────────▼──────────────┐
            │  runtime/policy_engine        │
            │   classify_query_intent       │
            │     read_only  → SIMPLE       │
            │     destructive → COMPLEX     │
            │     ambiguous   → LLM eval    │
            └───────────────┬──────────────┘
                            │
              ┌─────────────┴────────────┐
              ▼                          ▼
   ┌──────────────────┐        ┌─────────────────────┐
   │  Runtime Loop     │        │  HITL Pipeline      │
   │   context_budget  │        │   interrupt → wait  │
   │   stop_policy     │        │   for operator      │
   │   skill_catalog   │        │   decision → resume │
   │   tool_cache      │        └─────────┬───────────┘
   │   memory recall   │                  │
   └─────────┬─────────┘                  │
             │                            │
             ▼                            ▼
   ┌────────────────────────────────────────────────────┐
   │  integrations/clients/llm_engine — OllamaEngine     │
   │   [text protocol]   model emits [TOOL:name] {json}  │
   │   [native tools]    model emits structured tool_call│
   │                      → synthesized into [TOOL:] line│
   └─────────────────────────────┬──────────────────────┘
                                 │
                                 ▼
   ┌────────────────────────────────────────────────────┐
   │  integrations/router/tool_router — ToolRouter       │
   │   dispatches to: local callable / MCP / OpenAPI    │
   └────────────────────────────────────────────────────┘
```

Full dependency graph + module boundaries: see `ARCHITECTURE.md` §2.

---

## WebUI console

Open http://localhost:8000 — three-column layout:

| Column | Contents |
|---|---|
| **Left** | Skills catalog (clickable to inspect) + Tool registry (mock + pragmatic) |
| **Centre** | Chat (SSE streaming) + Flow tab (every turn's tool calls, stop reason) |
| **Right** | HITL approval cards + tool-result cache + memory recall |

**Tabs in the right pane**:
- **HITL** — pending approval cards (Approve / Reject / Edit args / Skip with note)
- **Cache** — large tool outputs (syslogs, Prometheus) stored externally with `[STORED:id]` references
- **Memory** — color-coded recall cards: facts (green), profile (blue), recent turns (gray)
- **Journal** — per-session skill load + tool call events for offline analysis

---

## HITL approval flow

Triggered when query involves destructive tools (`edit_device_config`, `restart_service`, `rollback_config`, ...) OR when the skill matcher is ambiguous OR when explicitly named in `cfg.hitl.tool_names`.

```
LLM emits [TOOL:edit_device_config] {device_id:"ap-01", ...}
              │
              ▼  intercepted by HitlExecutor before tool runs
        ChunkQueue.push(hitl_card)
              │
              ▼  WebUI right pane shows card
              │   ┌─────────────────────────────────┐
              │   │ Approve | Reject | Edit | Skip  │
              │   └─────────────────────────────────┘
              │
              ▼  operator clicks Approve
        POST /hitl/{id}/approve
              │
              ▼  HitlPipeline.resume(decision)
        Tool actually executes; result flows back into the loop
```

**Batch HITL**: `[TOOL_BATCH:edit_device_config] [{...}, {...}, ...]` opens one card per child target, operator approves each independently. SkillEvolver fires once at the end with the union of successful children.

Full details: `hitl_core/DESIGN.md`.

---

## Memory & learning loop

After every turn, six things happen automatically:

1. **FTS5 write** — turn (query + LLM response + tool calls) indexed for cross-session recall
2. **Fact extraction** — LLM extracts structured facts ("device ap-01 has IOS 15.4") to mid-term store
3. **Conflict reconciliation** — new facts go through `FactConflictDetector`: equivalent / refinement / contradiction / unrelated
4. **User model update** — expertise + traits (e.g. "operator prefers concise responses, uses CLI not GUI")
5. **Skill evolution** — `SkillJournalConsumer` watches dormant skills, fires `SkillEvolver.apply_feedback` to rewrite prompts
6. **Auto-consolidation** — every 30 turns per session, old chunks merge into LLM-summarised rollups so long sessions stay fast

Configure thresholds in `config.yaml`:

```yaml
memory:
  auto_consolidate_turns: 30           # 0 to disable
  consolidation_template: "structured" # or "legacy"

cross_module:
  journal_to_facts:
    enabled: true                       # promote journal observations to mid-term facts
  fact_conflict_detection:
    enabled: true
    llm_reconcile_enabled: false        # cheap heuristic only by default
```

Full memory architecture: `agent_memory/DESIGN.md`.

---

## CI & quality gates

Every PR is gated by `scripts/precheck.sh`:

```bash
./scripts/precheck.sh            # everything (audits + eval)
./scripts/precheck.sh --audits   # static audits only (~30s)
./scripts/precheck.sh --eval     # retrieval eval only
```

**6 static audits** (any FAIL → PR cannot merge):

| Audit | Catches |
|---|---|
| `syntax_sweep` | Any `.py` file with parse error |
| `audit_module_independence` | Cross-module import violations (e.g. `evaluation/` importing `runtime/`) |
| `audit_imports` | Import paths that don't resolve to any module |
| `audit_prompt_templates` | Unescaped `{...}` in f-string-like prompts that would crash at format time |
| `audit_directive_parsing` | `[TOOL:` parser bypass — single source of truth for tool-call extraction |
| `audit_wiring` | "Ghost services" — registered in `services[...]` but no external readers |

**Retrieval eval gate**:
- CI: BM25 backend, `recall@3 ≥ 0.40, MRR ≥ 0.30` against `data/golden_set.jsonl` (25 cases)
- Local: hybrid backend, `recall@3 ≥ 0.65, MRR ≥ 0.55`

**Tool-compliance bench** (`data/tool_compliance_set.jsonl`, 18 cases) — not in CI (needs running Ollama), runs locally or nightly:

```bash
# Baseline: text protocol
python -m evaluation.compliance_cli --golden data/tool_compliance_set.jsonl --model qwen2.5:7b

# Native tools (Ollama ≥ 0.4 + supported model)
python -m evaluation.compliance_cli --golden data/tool_compliance_set.jsonl --model qwen2.5:7b --native
```

Three independent metrics per case:
- `parse_ok` — model emitted valid `[TOOL:...]` syntax
- `name_ok` — model picked the right tool (or an acceptable alternative)
- `args_ok` — required args present, values match where pinned, no forbidden args

**pre-commit hook** (recommended): `pip install pre-commit && pre-commit install` — runs the same `--audits` locally before each commit.

**Branch protection** (GitHub): in repo Settings → Branches → require status checks for `Static audits`, `Production safety tests`, `Retrieval eval (BM25)`.

---

## Native tools (opt-in, Tier 1-C)

If you're running Ollama ≥ 0.4 with a tools-capable model (qwen2.5+, qwen3, llama3.1+, mistral-nemo, deepseek-v3, ...), flip the switch in `config.yaml`:

```yaml
llm:
  capabilities:
    supports_native_tools: true     # default false
```

Then restart. The engine ships an OpenAI-style `tools` array to Ollama and gets back **structured `tool_calls`** instead of free-text `[TOOL:name] {...}` directives. The engine synthesizes `[TOOL:name] {json}` lines from the structured response, so the runtime loop + directive parser + HITL flow are **completely unchanged** — this is a runtime upgrade, not an architectural change.

**What it fixes**: "model put device_id in the wrong field" / "model forgot a quote and JSON didn't parse" / "model hallucinated an extra arg" — these become structurally impossible because the model can't type the args by hand anymore; the API protocol requires them as a real dict.

**How to measure the improvement** on your specific model: run the compliance bench above with and without `--native`, compare `args_ok`.

**Rollback**: change one line of config + restart. No code changes.

---

## Operations cheatsheet

```bash
# Switch model
LLM_MODEL=qwen3.5:14b uvicorn main:app --port 8000

# Make a specific tool always require HITL
HITL_TOOL_NAMES=netflow_dump,db_failover uvicorn main:app

# Enable cross-module learning (journal → facts)
# In config.yaml:
#   cross_module:
#     journal_to_facts:
#       enabled: true

# Verbose LLM logs
LLM_LOG_DETAIL=compact LOG_MODE=llm uvicorn main:app --port 8000

# Run audits without commits
./scripts/precheck.sh --audits

# Run retrieval bench
python -m evaluation.cli --golden data/golden_set.jsonl --backend hybrid --top-k 5

# Run tool-compliance bench (needs Ollama)
python -m evaluation.compliance_cli --golden data/tool_compliance_set.jsonl --model qwen2.5:7b --verbose

# Check what's in memory for a session
sqlite3 data/memory/midterm.db "SELECT fact, fact_type, confidence FROM facts ORDER BY created_at DESC LIMIT 20"
```

---

## Project layout

```
NetOpYuAgent/
├── ARCHITECTURE.md            ← cross-module reference (read first for any multi-module change)
├── README.md / README-cn.md   ← onboarding (you are here)
├── main.py                    ← FastAPI app + lifespan; build_services() wires everything
├── config.py / config.yaml    ← all module activation knobs
│
├── runtime/                   ← agent loop, stop policy, directive parser, context budget
│   └── DESIGN.md
├── agent_memory/              ← 5-tier memory + consolidation + FTS5 + embedding hybrid
│   └── DESIGN.md
├── hitl_core/                 ← interrupt/decision/batch/audit pipeline
│   └── DESIGN.md
├── retrieval/                 ← BM25 / Embedding / Hybrid / Cache backends + meta tools
│   └── DESIGN.md
├── skills/                    ← catalog + journal + evolver + journal_consumer + loader
│   └── DESIGN.md
├── tools/                     ← mock + pragmatic tool implementations + metadata
│   └── DESIGN.md
├── integrations/              ← cross-module glue (LLM engine, MCP, OpenAPI, adapters)
│   └── DESIGN.md
│
├── schema/                    ← ArgSchema + JSON-Schema / Ollama tools exporter
├── evaluation/                ← retrieval bench + tool-compliance bench + CLIs
├── memory/                    ← thin MemoryAdapter facade for runtime
├── webui/                     ← FastAPI sub-app + SPA dashboard
├── a2a/                       ← agent-to-agent protocol primitives
├── task/                      ← task graph primitives
├── registry/                  ← agent identity + capability registry
│
├── data/
│   ├── golden_set.jsonl              ← retrieval bench (25 cases)
│   └── tool_compliance_set.jsonl     ← compliance bench (18 cases)
│
├── scripts/
│   ├── precheck.sh                   ← single entry for audits + eval (used by CI + pre-commit)
│   ├── audit_module_independence.py
│   ├── audit_imports.py
│   ├── audit_wiring.py
│   ├── audit_prompt_templates.py
│   ├── audit_directive_parsing.py
│   └── _audit_common.py
│
├── .github/workflows/ci.yml          ← 3 parallel jobs on every PR
└── .pre-commit-config.yaml           ← local audit gate before commit
```

For a full file-by-file breakdown of any module, read its `DESIGN.md`.

---

## Contributing

### Before opening a PR

```bash
# Set up the local quality gate (one-time):
pip install pre-commit
pre-commit install

# Run before pushing:
./scripts/precheck.sh
```

### Changing one module

Read **only that module's** `DESIGN.md`. Run tests; let CI catch the rest.

### Changing two modules

Read `ARCHITECTURE.md` §4 ("cross-module conventions"). Most cross-module wiring goes in `cfg.cross_module.*` — see `config.yaml`.

### Adding a new service (cross-module collaborator)

1. Construct it in `main.py:build_services()`
2. Register: `services["my_service"] = obj`
3. **Have at least one external file read it via `services.get("my_service")` or `services["my_service"]`** — otherwise `audit_wiring.py` flags it as a ghost service and CI fails
4. If it's introspection-only (no runtime caller — e.g. exposed via `/system/wiring`), add to `KEY_WHITELIST` in `audit_wiring.py` with justification

### Adding a tool-compliance case

Append a JSONL line to `data/tool_compliance_set.jsonl`:

```jsonl
{"query": "your query", "expected_tool": "tool_name", "expected_args": {"k": "v"}, "tags": ["destructive"]}
```

CI validates structure; bench measures actual model performance.

### Adding a retrieval golden case

Append to `data/golden_set.jsonl` (see existing cases for schema). Re-run `./scripts/precheck.sh --eval` to confirm thresholds still hold.

### Documentation rules

- Single-module change → update that module's `DESIGN.md`
- Cross-module change → update `ARCHITECTURE.md` if it touches the dependency graph
- New module → must ship a `DESIGN.md` with the 6 standard sections (this is a PR requirement)
- README is for onboarding only; depth belongs in `DESIGN.md` / `ARCHITECTURE.md`

---

## Roadmap

### Implemented

- ✅ A2A Protocol v0.3.0 — full SSE streaming + WebSocket HITL
- ✅ HITL pipeline (interrupt + 4 decision types + batch + audit) — `hitl_core/`
- ✅ 5-tier memory (short/mid/long/profile/journal) with FTS5 + embedding hybrid recall
- ✅ Fact conflict detection wired into `MemoryAdapter.add_fact` (semantic dedup + LLM reconcile)
- ✅ Auto memory consolidation every N turns per-session (background, non-blocking)
- ✅ Hermes post-turn pipeline (fact extraction, user model, skill evolver)
- ✅ MCP + OpenAPI tool backends; unified `ToolRouter`
- ✅ Skill catalog (L1/L2 progressive disclosure + composite scoring)
- ✅ SkillEvolver — autonomous creation + dormant-skill rewriting
- ✅ Tool result external storage (`[STORED:id]`) + paginated read API
- ✅ Context budget priority allocation (`cfg.context_budget.strategy=priority`)
- ✅ Stop policy six-dimension evaluation
- ✅ PolicyEngine intent classifier fast-path (read-only queries skip LLM eval)
- ✅ Trust-mode graduated HITL (cautious / standard / trusted)
- ✅ Lifecycle hooks (`runtime/hooks.py`) for extensibility without context cost
- ✅ JWT / API-key auth (`auth_core.py` / `auth.py`)
- ✅ Log redaction (secrets, API keys, SNMP community strings)
- ✅ Native tools API (Ollama OpenAI-style, opt-in via `supports_native_tools`)
- ✅ Tool-compliance bench (`evaluation/compliance_cli.py`) for A/B model evaluation
- ✅ Retrieval bench (`evaluation/cli.py`) with CI gate
- ✅ 6 static audits + CI + pre-commit + branch protection

### Reserved / not yet implemented

- ⏳ **Skill as sub-agent** — high-complexity skills (≥5 steps) become LangGraph subgraphs with independent prompt + budget. `cfg.skill_orchestration.subagent.*` namespace reserved.
- ⏳ **MemGPT-style LLM-managed memory** — agent self-manages tier promotion/demotion. Defer until fact corpus is reliably clean.
- ⏳ **OpenTelemetry tracing** — spans on all cross-module calls; `session_id` as TraceID.
- ⏳ **Postgres checkpointer** for HITL graph state (replaces in-memory `MemorySaver` for production durability).
- ⏳ **Distributed multi-agent A2A** — primitives exist (`a2a/`); orchestration layer pending.
- ⏳ **Per-tool circuit breakers** — partial in `ToolMeta`; UI integration pending.

---

## Glossary

| Term | Meaning |
|---|---|
| **Functional module** | Code unit with zero imports from other functional modules (`memory/`, `hitl_core/`, etc.). Verified by `audit_module_independence`. |
| **Adapter** | Opt-in bridge connecting two functional modules. Lives in `integrations/adapters/`. |
| **Skill** | A markdown procedure consulted by the LLM (not executable code). |
| **Tool** | An atomic callable — local fn / MCP server method / OpenAPI op. |
| **Meta-tool** | A tool the LLM uses to discover other tools (`list_tools`, `tool_details`). |
| **Journal** | Per-session log of skill loads + tool calls; consumed by `SkillEvolver`. |
| **Fact** | Structured statement extracted from conversation, stored in mid-term memory. |
| **HITL gate** | Mechanism pausing the agent until a human decides. |
| **Ghost service** | An object in `services[...]` with no external readers — `audit_wiring` flags these. |
| **Native tools** | Ollama's OpenAI-compatible `tools` API — structured tool_calls instead of text protocol. |
| **Trust mode** | `cautious` / `standard` / `trusted` — operator-set sensitivity for HITL gate triggering. |

---

## Environment variables

Common knobs (full list: see `config.py` field-by-field):

| Env var | Default | Effect |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` / `openai` / `anthropic` / `mock` |
| `LLM_MODEL` | `qwen3.5:27b` | Any Ollama tag or OpenAI/Anthropic model name |
| `LLM_BASE_URL` | `http://localhost:11434` | Override for remote Ollama |
| `LLM_LOG_DETAIL` | `summary` | `summary` / `compact` / `full` |
| `LLM_SUPPORTS_NATIVE_TOOLS` | `false` | Enable Ollama native tools API (Tier 1-C) |
| `HITL_BACKEND` | `core` | `core` (recommended) or `langgraph` (legacy) |
| `HITL_TOOL_NAMES` | (from config.yaml) | Comma-separated tool names to always gate |
| `HITL_SKILL_AMBIGUITY` | `false` | Gate when skill matcher confidence is low |
| `MEMORY_AUTO_CONSOLIDATE_TURNS` | `30` | 0 to disable per-session auto-consolidation |
| `MEMORY_CONSOLIDATION_TEMPLATE` | `structured` | `structured` (Hermes) or `legacy` |
| `DTM_COMPACTION_TURNS` | `5` | Flush daily `.md` after N turns (raise for prod) |
| `NETOPYU_JWT_SECRET` | (required if auth enabled) | HS256 signing key |
| `LOG_MODE` | `default` | `llm` / `verbose` / `default` / `quiet` |

---

## License

(see `LICENSE` in repo root)

---

*Version: v4.0 — May 2026 (post Tier 1-C / 2-E)*
*Last reviewed against: `ARCHITECTURE.md` rev 2026-05*