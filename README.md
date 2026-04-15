# IT Ops Multi-Agent Platform

> **IT 运维多智能体平台** — A2A-protocol multi-agent orchestration system with HITL human-in-the-loop, Hermes learning loop, dual-path execution engine, and a terminal-style WebUI console.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Module Reference](#3-module-reference)
4. [Quick Start](#4-quick-start)
5. [Running with a Real LLM (Ollama)](#5-running-with-a-real-llm-ollama)
6. [WebUI Console Guide](#6-webui-console-guide)
7. [HITL Human Approval Flow](#7-hitl-human-approval-flow)
8. [Hermes Learning Loop](#8-hermes-learning-loop)
9. [P0 Tool Result Cache](#9-p0-tool-result-cache)
10. [P1 / P2 Features](#10-p1--p2-features)
11. [Mock Tools & Skill Catalog](#11-mock-tools--skill-catalog)
12. [Environment Variables](#12-environment-variables)
13. [Project Structure](#13-project-structure)
14. [Roadmap](#14-roadmap)

---

## 1. Overview

An IT operations multi-agent platform built around two design principles:

**A2A Protocol First** — All agent communication uses Google's A2A Protocol over HTTP. Agents are discoverable, load-balanced, and health-checked through a built-in registry. Agents written in any language or framework can participate.

**Thin Loop, Thick Scaffold** — Inspired by Claude Code's architecture: the main execution loop (`AgentRuntimeLoop`) is deliberately thin. All the hard work lives in the scaffold around it — context budget management, tool result caching, stop policy, skill loading, FTS5 cross-session recall, and the Hermes post-turn learning pipeline.

### Core Capabilities

| Capability | Description |
|---|---|
| **Dual-path routing** | Simple queries → Runtime Loop (no LangGraph overhead). Complex/destructive operations → HITL LangGraph with human approval gate |
| **HITL approval** | LangGraph `interrupt()` pauses execution, browser shows approval card, graph resumes on decision |
| **Hermes learning loop** | After every turn: FTS5 memory write, MemoryCurator extracts facts, UserModelEngine tracks expertise, SkillEvolver creates reusable skill recipes |
| **FTS5 cross-session recall** | SQLite FTS5 stores all past turns; semantically similar context is injected into new queries automatically |
| **P0 tool result cache** | Large tool outputs (syslogs, Prometheus, NetFlow) are stored externally; prompt carries only a `[STORED:id]` reference |
| **Skill catalog** | Level 1 summaries injected every turn, Level 2 details loaded on demand — avoids spending tokens on irrelevant skills |
| **Context budget** | Per-turn token allocation: confirmed facts > working set > memory > tool results > environment. Soft cap 3 200 tokens |
| **Stop policy** | Six dimensions: max turns, max tool calls, token budget, low progress, low confidence, explicit stop signal |
| **Agent registry** | Runtime agent registration/deregistration, health checks, round-robin / random / least-loaded balancing |
| **MCP + OpenAPI** | Pluggable tool backends: MCP server (JSON config) and any OpenAPI 3.0 spec; mock mode for both |

---

## 2. Architecture

```
External caller (RouterAgent / WebUI / webhook / curl)
           │  A2A JSON-RPC over HTTP-SSE / REST
           ▼
┌──────────────────────────────────────────────────────────┐
│                      API gateway                          │
│   /api/v1/a2a/*   /hitl/*   /registry/*   /webui/*       │
└─────────────────────────┬────────────────────────────────┘
                          │
         ┌────────────────▼──────────────────┐
         │          Execution router          │
         │                                   │
         │  classify(query)                  │
         │    SIMPLE  → Runtime Loop         │
         │    COMPLEX → HITL Graph           │
         └───────────────────────────────────┘
              │                    │
   ┌──────────▼─────────┐  ┌──────▼──────────────────────┐
   │   Runtime Loop      │  │  HITL Graph (LangGraph)      │
   │   context_budget    │  │    intent_classifier         │
   │   stop_policy       │  │    risk_assessor             │
   │   skill_catalog     │  │    planner                   │
   │   tool_cache        │  │    hitl_interrupt_node ←─────│─── operator
   │   FTS5 recall       │  │    executor                  │
   └──────────┬──────────┘  │    result_formatter          │
              │             └──────────────────────────────┘
              ▼
   ┌──────────────────────────────────────────────────────┐
   │              Hermes post-turn pipeline                │
   │   FTS5SessionStore → MemoryCurator → UserModelEngine  │
   │                    → SkillEvolver                     │
   └──────────────────────────────────────────────────────┘
              │
   ┌──────────▼───────────────────────────────────────────┐
   │          Integrations layer                           │
   │   OllamaEngine / OpenAIEngine   MCP client           │
   │   OpenAPI client                ToolRouter            │
   └──────────────────────────────────────────────────────┘
```

### Request flow — SIMPLE path

`classify=SIMPLE` → FTS5 recall (past sessions) → `loop.stream()` with real LLM → Turn 1: LLM picks tool → tool executes (large output → `ToolResultStore`) → Turn 2: LLM synthesises answer → stop → Hermes pipeline fires (FTS5 write, curate, user model, skill evolver)

### Request flow — COMPLEX path

`classify=COMPLEX` → `executor.execute()` → `run_with_hitl()` → LangGraph streams → `hitl_interrupt_node` calls `interrupt()` → `register_interrupt()` → `TaskArtifactUpdateEvent` enqueued → SSE stream closes → browser shows HITL approval card → operator clicks Approve/Reject → `POST /hitl/{id}/approve` → `graph.ainvoke(None, thread_cfg)` resumes → executor node runs → result streams back

---

## 3. Module Reference

### 3.1 `runtime/` — Execution engine (8 files)

| File | Responsibility |
|---|---|
| `loop.py` | `AgentRuntimeLoop`: thin main loop — classify, stream, pre/post verify, tool dispatch |
| `context_budget.py` | `ContextBudgetManager`: per-turn token prioritisation; `ToolResultStore`: P0 large-output externalisation |
| `stop_policy.py` | `StopPolicy`: six-dimension stop evaluation; `LoopState`: cross-turn state tracking |
| `skill_catalog.py` | `SkillCatalogService` integration: L1 summary injection + L2 on-demand detail loading |
| `model_tier.py` | `ModelTierClassifier`: routes queries to fast model vs full model |
| `delegation.py` | Fork/fresh delegation modes, context inheritance policy |
| `tool_cache.py` | Composite skill scoring (tool overlap × 0.6 + semantic similarity × 0.4) |

### 3.2 `hitl/` — Human-in-the-loop (9 files)

| File | Responsibility |
|---|---|
| `a2a_integration.py` | `ITOpsHitlAgentExecutor`: dual-path router + post-turn verification hook |
| `graph.py` | LangGraph `StateGraph(ITOpsGraphState)`: 6 nodes (classify → risk → plan → interrupt → execute → format) |
| `triggers.py` | Four trigger types: destructive op / alert severity / confidence / ambiguous intent |
| `decision.py` | `HitlDecisionRouter`: approve / reject / edit / escalate / timeout |
| `review.py` | Five notification channels: Slack, PagerDuty, SSE, WebSocket, email (concurrent fan-out) |
| `audit.py` | Seven audit event types; in-memory and PostgreSQL backends |
| `router.py` | FastAPI routes: `/hitl/pending`, `/hitl/{id}/approve`, `/hitl/{id}/reject`, `/hitl/ws` |

**LangGraph state note:** `StateGraph(ITOpsGraphState)` uses a typed dict so LangGraph merges each node's return dict into accumulated state (`{**old, **new}`). Using a bare `dict` causes each node to receive only the previous node's output, silently dropping earlier keys like `intent_type` before `planner_node` runs.

### 3.3 `memory/` — Storage and learning (12 files)

| Layer | Backend | Retrieval | TTL |
|---|---|---|---|
| L1 in-process | Python list | full return | request lifetime |
| L2 short-term | Redis sorted set | time-descending | 24 h |
| L3 mid-term | ChromaDB vector index | cosine + time decay | 30 d |
| L4 long-term | PostgreSQL full-text | pg_trgm similarity | permanent |

**Hermes modules (active in this build):**

| Module | Role |
|---|---|
| `fts_store.py` | `FTS5SessionStore`: SQLite FTS5, cross-session turn search, FTS5-safe query sanitiser |
| `curator.py` | `MemoryCurator`: LLM-driven fact extraction from completed turns; system/user prompt separation prevents injection |
| `user_model.py` | `UserModelEngine`: tracks tool preferences, domain expertise, technical level, behavioral traits |

### 3.4 `skills/` — Skill system (3 files)

| File | Responsibility |
|---|---|
| `catalog.py` | `SkillCatalogService`: L1 summaries + L2 detail loading; composite scoring (tool overlap + semantic similarity) |
| `evolver.py` | `SkillEvolver`: autonomous skill creation after complex tasks; LLM decides reuse potential; feedback-driven self-improvement |

### 3.5 `integrations/` — LLM and tool backends (5 files)

| File | Responsibility |
|---|---|
| `llm_engine.py` | `OllamaEngine`, `OpenAIEngine`, `AnthropicEngine`, `MockEngine`; think-block stripping for reasoning models |
| `mcp_client.py` | MCP server client; JSON config or env-var driven; mock mode |
| `openapi_client.py` | OpenAPI 3.0 spec consumer; auto-generates tool definitions; mock mode |
| `tool_router.py` | `ToolRouter`: dispatches tool calls to MCP / OpenAPI / mock registry |

### 3.6 `task/` — Task orchestration (9 files)

- `intra/planner.py`: `TaskPlanner` (goal → DAG) + `TaskScheduler` (concurrency=5, dependency resolution) + `TaskExecutor`
- `inter/coordinator.py`: `A2ATaskDispatcher` (SSE delegation) + `MultiRoundCoordinator` (multi-turn context)
- `inter/hitl_bridge.py`: `HitlTaskBridge` (suspend / resume hooks)

### 3.7 `registry/` — Agent discovery (6 files)

- Dynamic AgentCard fetching (4 well-known paths)
- Health checks (60 s interval) + card refresh (300 s interval)
- Load balancing: `round_robin` (default) / `random` / `least_loaded`

### 3.8 `webui/` — Browser console

- `backend.py`: FastAPI sub-app mounted at `/webui`; `/chat/stream` SSE endpoint; `/hitl/*` approval endpoints; `/system/wiring` live wiring status
- `static/index.html`: terminal-style single-page console, no external JS framework; four tabs: Flow, HITL, Cache, Stats

---

## 4. Quick Start

### Requirements

- Python 3.9+ (3.12 recommended)
- [Ollama](https://ollama.ai) with `qwen3.5:27b` pulled (default LLM — see section 5)
- Optional: Redis, PostgreSQL, ChromaDB (auto-stub when absent)

### Install

```bash
cd NetOpYuAgent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # includes pyyaml for config.yaml loading
```

### Run (default: Ollama + qwen3.5:27b)

```bash
ollama serve
ollama pull qwen3.5:27b
uvicorn main:app --reload --port 8000
```

### Run (mock LLM — no Ollama needed)

```bash
# Override just the backend in config.yaml:
#   llm:
#     backend: "mock"
# Or via env var:
LLM_BACKEND=mock uvicorn main:app --reload --port 8000
```

Open:
- **WebUI console**: http://localhost:8000/webui/
- **A2A endpoint**: http://localhost:8000/api/v1/a2a/
- **HITL endpoints**: http://localhost:8000/hitl/
- **Registry**: http://localhost:8000/registry/
- **API docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

### Verify startup

The startup log prints a configuration summary:

```
━━ Configuration ━━
  LLM           : ollama/qwen3.5:27b  (base_url=http://localhost:11434)
  Tools         : MCP=mock  OpenAPI=mock
  HITL tools    : —
  Memory dir    : ./data
  DTM compaction: every 20 turns  nudge every 10 turns
  Log mode      : normal  LLM detail: off
  Server        : 0.0.0.0:8000  reload=True
```

The Stats tab → **🔌 System Wiring** panel shows a live green/red checklist. `GET /webui/system/wiring` returns JSON including DTM stats.

---

## 5. Running with a Real LLM (Ollama)

**Recommended:** set config in `config.yaml` (already defaults to Ollama + qwen3.5:27b):

```yaml
# config.yaml — already the default, no change needed
llm:
  backend:  "ollama"
  model:    "qwen3.5:27b"
  base_url: "http://localhost:11434"
```

Then simply:

```bash
ollama serve
ollama pull qwen3.5:27b
pip install -r requirements.txt   # includes pyyaml
uvicorn main:app --port 8000 --reload
```

**Alternative — env var override** (no config.yaml edit needed):

```bash
ollama serve
ollama pull qwen3.5:14b   # lighter model

LLM_MODEL=qwen3.5:14b uvicorn main:app --port 8000 --reload
```

Startup log confirms:
```
━━ Configuration ━━
  LLM           : ollama/qwen3.5:27b  (base_url=http://localhost:11434)
  Tools         : MCP=mock  OpenAPI=mock
  Memory dir    : ./data
  ...
```

**Supported backends:**

| `LLM_BACKEND` | Notes |
|---|---|
| `ollama` | Local Ollama server; default — set `llm.base_url` and `llm.model` in config.yaml |
| `openai` | OpenAI API; set `OPENAI_API_KEY` env var |
| `anthropic` | Anthropic API; set `ANTHROPIC_API_KEY` env var |
| `mock` | Deterministic stub, no LLM calls — for testing without Ollama |

Thinking models (e.g. `qwen3-coder`): the engine automatically strips `<think>…</think>` blocks before parsing tool calls, and passes `think=False` to the Ollama API.

---

## 6. WebUI Console Guide

Three-column layout:

### Left — Skills & Tools

- **Skills panel**: all registered skills; click a skill name to populate the query box. Green dot = low risk, amber = medium, red = high/critical. `HITL` badge = approval required before execution.
- **Quick Tools panel**: call any mock tool directly; large results appear in the Cache tab automatically.

### Centre — Chat

| Control | Description |
|---|---|
| Query box | Enter to send, Shift+Enter for newline |
| `session` | Leave blank to auto-generate; enter an ID to continue a previous session |
| `mode` | `stream` = SSE token-by-token; `sync` = wait for full response |
| `delegation` | `fresh` = independent sub-agent; `forked` = inherit parent agent's confirmed facts and working set |
| `[STORED:…]` chip | Click to load and page through a cached large result in the Cache tab |

**Flow tab**: every module event is streamed in real time — classify, FTS5 recall, pre-verify, tool calls, post-verify, Hermes curation, user model update, skill evolver. Click any row to expand details.

### Right — HITL / Cache / Stats

- **HITL tab**: pending interrupts with trigger type, risk level, proposed action, and Approve / Reject buttons. Polls `/hitl/pending` every 3 s; switches to this tab automatically when an interrupt fires.
- **Cache tab**: all `ToolResultStore` entries; page through large outputs with Prev/Next buttons.
- **Stats tab**: skill count, cache entries, active agents, Hermes module status, system wiring checklist.

---

## 7. HITL Human Approval Flow

### Trigger queries

```
restart the payments-service in production
rollback auth-service to version 3.2.1
drain k8s-worker-03 for maintenance
delete the staging database
force failover payments-db to replica
```

### What happens

1. `classify()` returns `COMPLEX` — logged as `Complexity: complex — Destructive action detected`
2. `executor.execute()` calls `run_with_hitl()` — LangGraph graph starts
3. `intent_classifier_node` → `risk_assessor_node` → `planner_node` → `route_after_plan()`
4. `DestructiveActionTrigger` fires → routes to `hitl_interrupt_node`
5. `interrupt(payload)` pauses the graph; checkpointer saves state
6. `_handle_interrupt_chunk()` calls `register_interrupt()` — payload now in `_payload_store`
7. `HitlA2AEventProcessor` emits `TaskArtifactUpdateEvent` — SSE delivers `hitl_interrupt` chunk to browser
8. Browser: `switchTab('hitl')` + `refreshHitl(5)` (5 retries × 500 ms)
9. HITL card appears: trigger, risk level, proposed action, SLA countdown
10. Operator clicks **Approve** → `POST /hitl/{id}/approve` → `graph.ainvoke(None, thread_cfg)` → executor node runs
11. Flow tab logs `⚠ HITL APPROVE` with outcome

### Approval decision types

| Decision | Effect |
|---|---|
| `approve` | Graph resumes, executor node runs the proposed action |
| `reject` | Graph routes to END, task marked rejected |
| `edit` | Operator patches proposed action params, then graph resumes |
| `escalate` | Escalates to a senior reviewer; SLA timer resets |
| `timeout` | Fires automatically when SLA expires |

### Debugging HITL

Key log lines to look for:

```
hitl.graph:    route_after_plan: intent_type='destructive_op' is_destructive=True action_type='restart_service'
hitl.graph:    HITL interrupt — interrupt_id=… trigger=destructive_op risk=high
hitl.graph:    Graph interrupt detection complete: found=True
hitl.decision: HITL registered: interrupt_id=… status=pending store_size=1
webui.backend: /hitl/pending: store_size=1 … returning 1 pending interrupts
```

If `found=False`: confirm `StateGraph(ITOpsGraphState)` is used (not bare `dict`), and that `intent_classifier_node` sees `"restart"` / `"rollback"` / `"delete"` / `"drain"` in the query.

---

## 8. Hermes Learning Loop

After every completed query, a post-turn pipeline runs automatically:

```
Turn completes
    │
    ├─→ FTS5SessionStore.write_turn()
    │       Writes user query + assistant response to SQLite FTS5
    │       Importance score based on tool usage and response length
    │
    ├─→ MemoryCurator.after_turn()
    │       LLM extracts structured facts from the turn
    │       Types: incident_lesson, config_fact, tool_preference,
    │              entity_relation, procedure_step, env_fact
    │       High-confidence facts stored in MemoryRouter (L2–L4)
    │
    ├─→ UserModelEngine.after_turn()
    │       Tracks: tool frequency, domain counts (auth / network /
    │               compute / storage), technical level, behavioral traits
    │       Injects user profile section into subsequent query prompts
    │
    └─→ SkillEvolver.after_task()   (COMPLEX tasks only)
            LLM asks: should this become a reusable skill?
            If yes: generates markdown skill recipe (steps, tools, risk)
            Stored in SkillCatalogService for future queries
```

**FTS5 cross-session recall** runs at the start of every query:

```
New query arrives
    → curator.recall_for_session(query, session_id)
    → FTS5 searches all past sessions for similar turns
    → Top matches injected as context before LLM call
    → Flow tab shows: 🔮 FTS5 Recall — N chars from M sessions
```

**WebUI events emitted by Hermes:**

| SSE chunk type | Meaning |
|---|---|
| `hermes_write` | Turn written to FTS5 |
| `hermes_curate` | `memories_count` facts extracted by MemoryCurator |
| `hermes_umodel` | User model updated — `technical_level`, `domain_counts`, `trait_count` |
| `hermes_skill` | Skill created or updated by SkillEvolver (COMPLEX only) |
| `recall` | Past context injected — shows `chars` and `sessions_searched` |

**Data directory:** `HERMES_DATA_DIR=./data` (default). Contains `state.db` (FTS5 turns) and any skill recipe files written by SkillEvolver.

---

## 9. P0 Tool Result Cache

### Problem

IT ops tools can return tens of thousands of bytes — 300-line syslogs, 60-minute Prometheus time series, 500-record NetFlow dumps. Injecting these directly into the prompt exhausts the context window on the first tool call.

### Solution

`ToolResultStore` + `ContextBudgetManager` — two-step externalisation:

1. Tool output > 4 000 chars → stored in `ToolResultStore`. Prompt receives only:
   ```
   [STORED:syslog_search:a3f9c12b] Preview: Apr 10 09:12:01 ap-01 hostapd…
   ```

2. LLM reads details on demand via `read_stored_result`:
   ```
   [TOOL:read_stored_result] {"ref_id": "a3f9c12b", "offset": 0, "length": 2000}
   ```

### Walkthrough

```
1. Query: "search syslogs for errors on ap-01"
   → Runtime Loop calls syslog_search (~6 000 chars)
   → Stored automatically; [STORED:…] reference injected into prompt
   → Cache tab shows new entry

2. Click [STORED:syslog_search:abc123] chip in chat
   → Cache tab loads first page (2 000 chars)
   → Shows: Total: 6XXX chars | Has more: True | Next offset: 2000

3. Click "Next ▶" to page through remaining content
```

### API

```bash
# Trigger a large-result tool
curl -X POST http://localhost:8000/webui/tools/syslog_search \
  -H "Content-Type: application/json" \
  -d '{"args": {"host": "ap-01", "keyword": "error", "lines": 300}}'

# Read a stored result (paginated)
curl "http://localhost:8000/webui/tools/result/{ref_id}?offset=0&length=2000"
```

---

## 10. P1 / P2 Features

### P1: Pre- and post-verification

```python
# Pre-verify: runs before execution, blocks destructive ops
pre = await loop.pre_verify(query, confirmed_facts, env_context)
if not pre.passed:
    return STOP_HITL  # escalates to HITL graph

# Post-verify: runs after each tool call
post = await loop.post_verify(tool_name, result, confirmed_facts)
if not post.passed:
    state.unresolved_points.append(f"Post-verify: {post.reason}")
```

Built-in rules: destructive operations always fail pre-verify; closed change window + change op fails; `allow_destructive=False` + production env fails.

### P1: Confirmed Facts & Working Set

```python
# Confirmed facts — injected at highest priority in every prompt
state.record_new_fact("payments-service is healthy in prod")
state.record_new_fact("DNS resolution confirmed OK; not the cause")

# Working set — currently focused devices
working_set = [
    DeviceRef(id="ap-01", label="AP-01 at Site-A"),
    DeviceRef(id="sw-core-01", label="Core Switch"),
]
```

### P1: Forked delegation

```python
# fresh: independent sub-agent, starts from scratch
# forked: inherits parent's confirmed facts and working set
delegation = "forked"
```

### P2: Model tiering

```python
decision = loop.classify(query)
print(decision.model_tier)   # "fast_model" or "full_model"

# fast_model: check / status / dns / list queries
# full_model: complex analysis, P0/P1 events, destructive ops
# In production: map to different Ollama models or haiku vs sonnet
```

---

## 11. Mock Tools & Skill Catalog

### Inventory tools (use these for "what devices exist?" queries)

```bash
POST /webui/tools/list_devices
{"args": {}}                                      # all 13 devices (APs, switches, routers, servers)
{"args": {"type": "switch"}}                       # wired switches only
{"args": {"type": "wireless_ap"}}                  # wireless APs only
{"args": {"type": "router"}}                       # routers only
{"args": {"type": "switch", "site": "site-a"}}    # site-filtered

POST /webui/tools/list_interfaces
{"args": {"device_id": "sw-core-01"}}             # port table for a switch
{"args": {"device_id": "ap-01"}}                  # radio interfaces for an AP
{"args": {"device_id": "router-01"}}              # WAN/LAN interfaces for a router
```

The 13 mock devices span: 4 wireless APs, 5 switches (2 core + 3 access), 2 edge routers, 2 RADIUS servers — all at site-a or site-b.

### Large-result tools (trigger P0 cache)

```bash
POST /webui/tools/syslog_search
{"args": {"host": "ap-01", "keyword": "error", "lines": 300}}    # ~6 KB

POST /webui/tools/prometheus_query
{"args": {"metric": "up", "job": "network_devices", "range_minutes": 60}}  # ~5 KB

POST /webui/tools/netflow_dump
{"args": {"site": "site-a", "flows": 500}}   # ~10 KB
```

### Small-result tools (inline)

```bash
POST /webui/tools/dns_lookup      {"args": {"hostname": "payments.internal"}}
POST /webui/tools/device_info     {"args": {"device_id": "sw-core-01"}}
POST /webui/tools/alert_summary   {"args": {"severity": "P1"}}
POST /webui/tools/service_health  {"args": {"service": "payments-service"}}
```

### Cache tools

```bash
# Read a page of a stored large result
POST /webui/tools/read_stored_result
{"args": {"ref_id": "a3f9c12b", "offset": 0, "length": 2000}}

# Process all chunks of a stored result with an operation
POST /webui/tools/process_stored_chunks
{"args": {"ref_id": "a3f9c12b", "operation": "filter", "match": "ERROR"}}
{"args": {"ref_id": "a3f9c12b", "operation": "extract", "pattern": "(\\d+\\.\\d+\\.\\d+\\.\\d+)"}}
{"args": {"ref_id": "a3f9c12b", "operation": "count",   "match": "timeout"}}
{"args": {"ref_id": "a3f9c12b", "operation": "summarise"}}
```

### Skill catalog (9 pre-built skills)

| Skill ID | Risk | HITL |
|---|---|---|
| `radius_auth_diagnosis` | low | no |
| `bgp_neighbor_check` | low | no |
| `dns_resolution_debug` | low | no |
| `k8s_pod_restart` | high | **yes** |
| `db_failover` | critical | **yes** |
| `syslog_bulk_analysis` | low | no |
| `network_traffic_analysis` | low | no |
| `prometheus_alert_triage` | medium | no |
| `change_window_check` | low | no |

---

## 12. Configuration

All settings live in **`config.yaml`** (project root). Edit that file to change defaults — no code changes needed.

Environment variables always override the YAML value (12-factor compatible).

### Quick-start config

```yaml
# config.yaml — change these for your deployment
llm:
  backend: "ollama"        # ollama | openai | anthropic | mock
  model:   "qwen3.5:27b"  # any Ollama-compatible model
  base_url: "http://localhost:11434"

logging:
  mode: "normal"           # normal | llm | verbose
```

### Full environment variable reference

#### LLM

| Variable | YAML key | Default | Description |
|---|---|---|---|
| `LLM_BACKEND` | `llm.backend` | `ollama` | `ollama` \| `openai` \| `anthropic` \| `mock` |
| `LLM_MODEL` | `llm.model` | `qwen3.5:27b` | Model name passed to the backend |
| `LLM_BASE_URL` | `llm.base_url` | `http://localhost:11434` | Ollama or OpenAI-compatible base URL |
| `LLM_TEMPERATURE` | `llm.temperature` | `0.1` | Sampling temperature |
| `LLM_MAX_TOKENS` | `llm.max_tokens` | `2048` | Max tokens per LLM call |
| `LLM_LOG_DETAIL` | `llm.log_detail` | `off` | `off` \| `compact` \| `full` — show LLM conversation in server log |

#### Logging

| Variable | YAML key | Default | Description |
|---|---|---|---|
| `LOG_MODE` | `logging.mode` | `normal` | `normal` — INFO only \| `llm` — DEBUG for LLM/tool interactions \| `verbose` — all DEBUG |

#### Tools

| Variable | YAML key | Default | Description |
|---|---|---|---|
| `MCP_USE_MOCK` | `tools.mcp.use_mock` | `true` | Use built-in NetOps mock instead of real MCP server |
| `MCP_CONFIG_JSON` | `tools.mcp.config_json` | — | MCP server config (JSON string or path to JSON file) |
| `OPENAPI_USE_MOCK` | `tools.openapi.use_mock` | `true` | Use built-in NetOps mock instead of real OpenAPI spec |
| `OPENAPI_SPEC_URL` | `tools.openapi.spec_url` | — | URL to OpenAPI 3.0 spec |
| `OPENAPI_BASE_URL` | `tools.openapi.base_url` | — | Base URL for OpenAPI calls |
| `OPENAPI_AUTH_TYPE` | `tools.openapi.auth_type` | `bearer` | `bearer` \| `api_key` \| `none` |
| `OPENAPI_TOKEN_ENV` | `tools.openapi.token_env` | `NETOPS_API_TOKEN` | Env var holding the API token |
| `HITL_TOOL_NAMES` | `tools.hitl_tool_names` | — | Comma-separated tools that always require HITL approval before execution |

#### HITL

| Variable | YAML key | Default | Description |
|---|---|---|---|
| `HITL_CONFIDENCE_THRESHOLD` | `hitl.confidence_threshold` | `0.75` | Confidence below this triggers HITL ambiguity check |
| `HITL_MAX_AUTO_HOST_COUNT` | `hitl.max_auto_host_count` | `5` | Actions affecting more hosts than this trigger HITL |
| `HITL_SKILL_AMBIGUITY` | `hitl.skill_ambiguity` | `false` | `true` to route ambiguous skill matches to HITL approval |
| `HITL_SLACK_WEBHOOK_URL` | `hitl.slack_webhook_url` | — | Slack incoming webhook URL for HITL notifications |
| `HITL_PAGERDUTY_ROUTING_KEY` | `hitl.pagerduty_routing_key` | — | PagerDuty Events API routing key |

#### Memory / Dual-Track Memory (DTM)

| Variable | YAML key | Default | Description |
|---|---|---|---|
| `HERMES_DATA_DIR` | `memory.data_dir` | `./data` | Root directory for FTS5 database, daily .md files, facts.jsonl, skills |
| `REDIS_URL` | `memory.redis_url` | — | Redis connection string (stubs to in-memory if absent) |
| `POSTGRES_DSN` | `memory.postgres_dsn` | — | PostgreSQL DSN (skips persistence if absent) |
| `CHROMA_PATH` | `memory.chroma_path` | `./chroma_db` | ChromaDB local path |
| `DTM_COMPACTION_TURNS` | `memory.dtm.compaction_turns` | `20` | Turns before flushing today's buffer to `daily/YYYY-MM-DD.md` |
| `DTM_NUDGE_TURNS` | `memory.dtm.nudge_turns` | `10` | Turns between deep Hermes LLM review sweeps |
| `DTM_TRACK_B_WEIGHT` | `memory.dtm.track_b_weight` | `1.5` | Score multiplier for curated facts vs raw chunks (>1 prefers facts) |
| `DTM_HALF_LIFE_DAYS` | `memory.dtm.temporal_half_life_days` | `7.0` | Track A temporal decay — score halves every N days |

#### Registry

| Variable | YAML key | Default | Description |
|---|---|---|---|
| `AGENT_URLS` | `registry.agent_urls` | — | Comma-separated peer agent A2A URLs for pre-population |
| `REGISTRY_LB` | `registry.lb_strategy` | `round_robin` | `round_robin` \| `random` \| `least_loaded` |
| `REGISTRY_HEALTH_INTERVAL` | `registry.health_check_interval` | `60` | Health check interval in seconds |

#### Server & A2A

| Variable | YAML key | Default | Description |
|---|---|---|---|
| `HOST` | `server.host` | `0.0.0.0` | Bind address |
| `PORT` | `server.port` | `8000` | Listen port |
| `RELOAD` | `server.reload` | `false` | Enable hot reload |
| `A2A_BASE_URL` | `server.a2a_base_url` | `http://localhost:8000/api/v1/a2a` | This agent's outbound A2A base URL |

### Override examples

```bash
# Run with a smaller model for testing
LLM_MODEL=qwen3.5:14b uvicorn main:app --port 8000

# Enable full LLM conversation logging
LLM_LOG_DETAIL=compact LOG_MODE=llm uvicorn main:app --port 8000

# Enable HITL for specific tools + skill ambiguity
HITL_TOOL_NAMES=netflow_dump,db_failover HITL_SKILL_AMBIGUITY=true uvicorn main:app

# Faster DTM compaction for development (see daily files appear sooner)
DTM_COMPACTION_TURNS=3 uvicorn main:app --port 8000

# Use real Ollama (same as config.yaml defaults)
LLM_BACKEND=ollama LLM_MODEL=qwen3.5:27b uvicorn main:app --port 8000
```

---

## 13. Project Structure

```
it-ops-agent/
├── main.py                        # FastAPI entry point — config-driven service assembly (v5)
├── config.yaml                    # Single source of truth for all settings
├── config.py                      # YAML loader + env var overlay + AppConfig dataclass
├── logging_config.py              # Logging presets: normal | llm | verbose
├── requirements.txt
├── pytest.ini
│
├── a2a/                           # A2A protocol (9 files)
│   ├── schemas.py                 # Pydantic data models
│   ├── agent_card.py              # AgentCard builder
│   ├── agent_executor.py          # Executor base class + processor chain
│   ├── event_queue.py             # Sealed async event queue
│   ├── request_handler.py         # JSON-RPC method router
│   ├── server.py                  # FastAPI sub-app factory
│   ├── push_notifications.py      # Push notifications (exponential backoff)
│   └── task_store.py              # In-memory task state store
│
├── hitl/                          # Human-in-the-loop (9 files)
│   ├── schemas.py                 # HitlPayload, HitlDecision, AuditRecord …
│   ├── triggers.py                # Four trigger types + HitlConfig
│   ├── graph.py                   # LangGraph StateGraph(ITOpsGraphState), 6 nodes
│   ├── review.py                  # Five notification channels + WebSocket manager
│   ├── decision.py                # HitlDecisionRouter — five decision types
│   ├── audit.py                   # Audit service (in-memory + PostgreSQL)
│   ├── router.py                  # FastAPI routes + /ws WebSocket endpoint
│   └── a2a_integration.py         # ITOpsHitlAgentExecutor — dual-path + post-verify
│
├── memory/                        # Memory and learning (12 files)
│   ├── schemas.py                 # MemoryRecord, RetrievalQuery …
│   ├── router.py                  # MemoryRouter façade
│   ├── fts_store.py               # FTS5SessionStore — SQLite FTS5 cross-session recall
│   ├── curator.py                 # MemoryCurator — LLM-driven fact extraction
│   ├── user_model.py              # UserModelEngine — expertise and trait tracking
│   ├── consolidation.py           # Consolidation worker (summary + entity extraction)
│   ├── stores/backends.py         # L1–L4 store implementations
│   └── pipelines/                 # ingestion.py + retrieval.py
│
├── skills/                        # Skill system (3 files)
│   ├── catalog.py                 # SkillCatalogService + 9 default skills
│   └── evolver.py                 # SkillEvolver — autonomous skill creation + self-improvement
│
├── integrations/                  # LLM and tool backends (5 files)
│   ├── llm_engine.py              # Ollama / OpenAI / Anthropic / Mock engines
│   ├── mcp_client.py              # MCP server client
│   ├── openapi_client.py          # OpenAPI 3.0 consumer
│   └── tool_router.py             # ToolRouter — dispatches to MCP / OpenAPI / mock
│
├── runtime/                       # Execution engine (8 files)
│   ├── loop.py                    # AgentRuntimeLoop — thin main loop
│   ├── context_budget.py          # ContextBudgetManager + ToolResultStore (P0)
│   ├── stop_policy.py             # StopPolicy + LoopState
│   ├── skill_catalog.py           # SkillCatalogService integration
│   ├── model_tier.py              # ModelTierClassifier (P2)
│   ├── delegation.py              # Fork / fresh delegation
│   └── tool_cache.py              # Composite skill scoring
│
├── task/                          # Task orchestration (9 files)
│   ├── schemas.py                 # TaskDefinition, SessionRecord …
│   ├── intra/planner.py           # TaskPlanner + TaskScheduler + TaskExecutor
│   ├── intra/store.py             # TaskStore + RetryManager
│   ├── inter/coordinator.py       # A2ATaskDispatcher + MultiRoundCoordinator
│   ├── inter/session.py           # SessionManager (Redis, TTL=8 h)
│   └── inter/hitl_bridge.py       # HitlTaskBridge — suspend / resume hooks
│
├── registry/                      # Agent registry (6 files)
│   ├── schemas.py                 # AgentEntry, AgentSkill, ResolutionResult
│   ├── store.py                   # InMemory + Redis dual storage
│   ├── discovery.py               # AgentDiscovery — AgentCard fetching
│   ├── registry.py                # AgentRegistry — register / resolve / health check
│   └── router.py                  # FastAPI routes
│
├── tools/                         # Mock tools (2 files)
│   └── mock_tools.py              # 11 mock tools: list_devices, list_interfaces, syslog_search, prometheus_query, netflow_dump, dns_lookup, device_info, alert_summary, service_health, read_stored_result, process_stored_chunks
│
├── webui/                         # Browser console
│   ├── backend.py                 # FastAPI sub-app — chat/stream, HITL, system wiring
│   └── static/index.html          # Terminal-style single-page console
│
└── tests/                         # Test suite
    ├── test_a2a.py                # A2A module (~30 cases)
    ├── test_hitl.py               # HITL module (~25 cases)
    ├── test_memory_task.py        # Memory + Task (~45 cases)
    ├── test_registry.py           # Registry (~30 cases)
    ├── test_runtime.py            # Runtime Loop (~50 cases)
    ├── test_hermes_features.py    # Hermes learning loop (~43 cases)
    └── test_p0_p1_p2.py           # P0/P1/P2 + WebUI (~60 cases)
```

---

## 14. Roadmap

### Implemented

- [x] A2A Protocol v0.3.0 — full SSE streaming + WebSocket HITL
- [x] HITL five-layer architecture (trigger → graph interrupt → notification fan-out → decision routing → audit)
- [x] `StateGraph(ITOpsGraphState)` — typed state prevents silent key clobbering between nodes
- [x] Four-layer memory (L1–L4, MMR retrieval, consolidation worker)
- [x] Hermes post-turn pipeline (FTS5 recall, MemoryCurator, UserModelEngine, SkillEvolver)
- [x] Integrations — Ollama / OpenAI / Anthropic / MCP / OpenAPI tool backends
- [x] Task module (DAG scheduling + cross-agent delegation + multi-turn sessions)
- [x] Registry — dynamic service discovery, health checks, load balancing
- [x] Runtime Loop — thin main loop + dual-path routing + P1/P2 features
- [x] P0: ToolResultStore large-output externalisation + paginated read API
- [x] P0: ContextBudgetManager per-turn token prioritisation
- [x] P0: StopPolicy six-dimension stop evaluation
- [x] P1: SkillCatalogService — L1/L2 progressive disclosure + composite scoring
- [x] P1: SkillEvolver — autonomous skill creation and feedback-driven improvement
- [x] P1: Forked delegation — fresh / forked + context inheritance
- [x] P1: Confirmed Facts & Working Set as first-class prompt citizens
- [x] P1: Pre- and post-verification hooks
- [x] P2: Model tiering — fast_model / full_model routing
- [x] WebUI console — terminal style, SSE/sync modes, Flow tab, HITL approval, P0 visualisation

### Production requirements (not yet implemented)

- [ ] **JWT / OIDC auth**: `/hitl/{id}/approve` and `/hitl/{id}/reject` must be restricted to approver roles — highest priority
- [ ] **Real LLM chains in graph nodes**: replace keyword stubs in `intent_classifier_node`, `risk_assessor_node`, `planner_node` with structured LLM calls
- [ ] **Real tool backends**: replace `_execute_task` stubs in `task/intra/planner.py` (kubectl / Prometheus HTTP API / OpsGenie)
- [ ] **Real embeddings**: replace `_EmbedderStub` in `memory/pipelines/ingestion.py`
- [ ] **OpenTelemetry tracing**: add spans to all cross-module calls; propagate TraceID via `session_id`
- [ ] **PostgreSQL checkpointer**: replace `MemorySaver` in `build_hitl_graph()` for production durability

### Nice-to-have

- [ ] Prompt cache-friendly prefix ordering (stable prefix first, variable content last)
- [ ] Lightweight verifier agent (small model for post-execution health checks)
- [ ] Agent versioning (AgentCard `version` field, canary releases)
- [ ] MCP protocol integration for external tool access

---

*Version: v3.0 | Updated: 2026-04-15*