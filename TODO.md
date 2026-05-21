# TODO — Production Readiness & Roadmap Tracker

> Tracking doc for deferred work. **Rule: when an item is done, delete it from this file.**
> Items that remain are not yet done. Last updated: 2026-05.

This file tracks the production-readiness items surfaced in the Sprint-3
readiness assessment (see `ARCHITECTURE.md §8.4` "What Sprint-3-pre does NOT fix")
plus the multi-agent roadmap (`ARCHITECTURE.md §11.3`).

---

## Sprint 3 — Production readiness

Grouped by tier. Each item has: effort estimate, what it fixes, and acceptance criteria.

### Tier A — Security blockers (do before ANY internet-facing deploy)

#### A1 — Auth-required-in-production startup check
- **Effort**: 0.5 day
- **Why**: `cfg.auth.enabled=False` + `ENVIRONMENT=production` currently boots fine. An operator who forgets to flip auth ships an open API where anyone can approve destructive HITL actions.
- **Acceptance**: Boot raises `RuntimeError` if `ENVIRONMENT=production` and `auth.enabled=false`, OR if `auth.enabled=true` but the JWT secret env var is unset.

#### A2 — CSRF protection on `/hitl/*` POST routes
- **Effort**: 1 day
- **Why**: A logged-in operator visiting a malicious page could have it POST to `/hitl/{id}/approve` and silently approve destructive ops.
- **Acceptance**: HITL approval endpoints reject requests without a valid CSRF token (or require `X-Requested-With: XMLHttpRequest` as a minimum bar).

#### A3 — Secrets management
- **Effort**: 2-3 days
- **Why**: Device passwords live in env vars; `kubectl describe pod` exposes them. Need Vault / AWS Secrets Manager / k8s Secrets with `secretRef`.
- **Acceptance**: `cfg.secrets.backend` selector (env / vault / aws_sm); a `secret_resolver.py` resolves `${...}` placeholders at startup; secrets never written to logs.

### Tier B — Deployability

#### B1 — Dockerfile + docker-compose
- **Effort**: 1 day
- **Why**: No container artifacts; ops can't deploy. `uvicorn main:app` is a dev command, not production.
- **Acceptance**: `Dockerfile` (non-root user, healthcheck, gunicorn+uvicorn worker); `docker-compose.yml` with ollama + redis services; `.env.production.example`.

#### B2 — `/livez` + `/readyz` endpoints
- **Effort**: 1 day
- **Why**: `/health` returns 200 even when Ollama is dead, so k8s liveness probes can't detect a broken agent — requests just hang for 5 min.
- **Acceptance**: `/livez` = process alive (cheap). `/readyz` = checks Ollama reachable + memory DB writable + HITL store writable; returns 503 when any dependency is down.

#### B3 — DEPLOYMENT.md operations runbook
- **Effort**: 1 day
- **Why**: All current docs are dev-facing. An SRE has no resource sizing / upgrade procedure / disaster-recovery runbook.
- **Acceptance**: `docs/DEPLOYMENT.md` covering external service deps, min CPU/RAM/disk/GPU, upgrade flow (not losing HITL pending state), monitoring alert thresholds.

### Tier C — Observability + reliability

#### C3 — Backup cron + restore runbook
- **Effort**: 1 day
- **Why**: SQLite corrupt (power loss / disk full / volume delete) loses all facts + skill history + audit log.
- **Acceptance**: `scripts/backup.sh` (sqlite `.backup` dump, nightly, rotate 30 days, optional S3 upload); restore steps documented in DEPLOYMENT.md.

#### C4 — Database migration framework
- **Effort**: 2 days
- **Why**: SQLite schema changes (add column, change type) crash on old data or silently ignore it. No versioned migrations.
- **Acceptance**: Alembic or a simple `migrations/` dir + `schema_version` table + serial apply at startup.

### Tier D — Performance / scaling

(none remaining — D1 done)

---

## Multi-agent roadmap

### Phase 2A — Business profile decoupling + role isolation ✅ DONE (2026-05)
Delivered: `profiles/` layer (default/lan/dc), `AGENT_PROFILE` selector,
`ToolLoader`/`SkillLoader` profile-aware, LAN tools migrated out of
`tools/mock_tools.py`, new DC fabric tools, `audit_profiles.py`,
`tests/test_profiles.py` (20 tests). Two role-isolated agents can run locally;
their tool sets are disjoint (verified by audit). See ARCHITECTURE.md §12.
*(Kept here until a release cycle, then delete.)*

### Phase 2A.1 — Per-agent data isolation ✅ DONE (2026-05)
Delivered: `cfg.agent_data_dir()` resolves `data/agents/<agent_id>/`;
memory / tool_results / hitl_checkpoints / evolved-skills all routed under it;
shared fixtures (golden_set, tool_compliance_set) stay at `data/`;
`scripts/migrate_data_to_agent.sh` migrates legacy single-agent state.
See ARCHITECTURE.md §12.7. *(Kept until a release cycle, then delete.)*

#### Phase 2A.1 follow-up — skill journal persist path not auto-isolated
- **Effort**: 15 min
- **Why**: `skill_orchestration.journal_persist_path` is operator-set and
  defaults to off (in-memory only). If an operator sets it to a literal path,
  two agents would share one journal file. Low risk (off by default).
- **Action**: when an operator enables journal persistence for a multi-agent
  deployment, document that they should make the path agent-specific (e.g.
  include `${AGENT_ID}`), or route it through `agent_data_dir()` too.

#### Phase 2A follow-up — pragmatic mode not profile-split
- **Effort**: 1-2 days
- **Why**: `tools/pragmatic_tools.py` (real Netmiko/NAPALM device tools) loads
  regardless of profile. A `dc` agent in pragmatic mode would still get the
  LAN device tools. mock mode (what A2A validation uses) IS split correctly.
- **Action**: when pragmatic multi-domain becomes real, split
  `pragmatic_tools.py` into `profiles/<id>/pragmatic_tools.py` and have
  ToolLoader pragmatic branch read from the profile too.

#### Phase 2A follow-up — per-profile MCP / OpenAPI integrations
- **Effort**: 1 day
- **Status**: partially handled (2026-05). The built-in `netops` MCP + OpenAPI
  mock now load ONLY for `profile=lan` (or pragmatic). DC/default get none.
- **Why remaining**: when DC needs its own MCP servers / OpenAPI specs, the
  current code has no per-profile integration config — it's a binary lan-or-not
  gate. Generalize to a `Profile.integrations` declaration (list of MCP server
  configs + OpenAPI specs per profile) so each domain wires its own.

### Phase 2B — Capability-based delegation
- **Effort**: 5-7 days
- **Status**: **IN PROGRESS (2026-05)**. Design reviewed + approved
  (PHASE_2B_DESIGN.md). Decisions: explicit `[DELEGATE:agent_id]` only (no
  auto-delegate yet); default fresh (facts NOT shared, `#forked` opt-in);
  reuse registry `_pick` for `*capability`, direct lookup for explicit
  agent_id; entry agent audits delegation boundary only; `[DELEGATE:]` is
  MUTUALLY EXCLUSIVE with `[TOOL:]` in one turn.
- **Scope**: `runtime/directive_parser.py` adds `[DELEGATE:]` regex;
  `runtime/loop.py` gets a `_delegate_to_peer()` branch; fix
  `A2ATaskDispatcher` URL bug (posts to `/stream`, server exposes `POST /`);
  wire `build_task_services()` into main.py (currently never called); chunks
  tagged `source_agent` so WebUI shows "via dc-agent". Plus debt #10
  (action_type builder) + #7/#12-3 (build_resumption_query).
- **Explicitly NOT in 2B**: cross-agent HITL passthrough (Phase 3), multi-hop,
  auto-delegate, parallel fan-out, cross-agent memory writeback.

### HITL design debt (reviewed 2026-05 against current tree)

Several items from the HITL debt audit were ALREADY FIXED in the H2 hardening
pass (claim_async_pending race fix #5; unified SLA watchdog #3 in-process half;
inject-queue bounds #6; per-agent hitl db #11; audit Check 5 path #12-2).
Remaining real debt, by priority:

#### HITL-P0 — async-HITL state persistence (debt #1 + #2)
- **Effort**: 3-4 days (declarative resumer handlers) + 2 days (Redis SSE broker)
- **Why**: `_async_registry` / `_session_sse_emit` are per-process in-memory.
  Single-process dev/prod is fine; uvicorn `--workers N` or a restart loses
  pending async-HITL callbacks (operator approve → router finds no pending →
  silent drop). Delegation does NOT make this worse (it uses the persistent
  A2A task store, not these dicts — see PHASE_2B_DESIGN §6), but horizontal
  scale of the HITL feature itself needs this.
- **Action**: serialize pending via resumer_name handler registry (callbacks
  are closures, can't pickle — register declarative handlers like ResumeHandle
  already does); SSE via Redis pub-sub or sticky sessions.
- **Interim** (cheap): on startup, sweep store for PENDING+ASYNC_NONBLOCKING
  past SLA → mark EXPIRED + audit + UI "lost on restart" hint.

#### HITL-P1 — follow-up turn UX + context (debt #4 + #7-residual)
- **Effort**: #4 follow-up UI loading 1.5d; #7 resumption-query is being done
  in Phase 2B (build_resumption_query shared with delegation).
- **#4**: operator approve → follow-up turn blocks HTTP 10-60s with no loading
  state. Fix: return `{async_followup_pending}` immediately + poll, or reopen SSE.

#### HITL-P1 — demo autoresponder config flag (debt #8)
- **Effort**: 0.5 day, best done WITH HITL-P2 #9.
- **Why**: `_demo_autoreply` is a tool arg; production has no clean off switch.
- **Action**: `config.yaml agent.h2.demo_mode` (dev true / prod false); tool
  reads config not arg. SLA watchdog already prevents the leak when off.

#### HITL-P2 — framework-ize H2 via PolicyEngine hitl_mode (debt #9 + #12-1 + #8)
- **Effort**: 3-4 days. Suggest as its own **Phase 2C**.
- **Why**: `tool_meta` marks `hitl_mode: async_nonblocking` but PolicyEngine
  ignores it — H2 is hand-coded in query_radius_logs. Ideal: PolicyEngine reads
  the metadata → runtime wraps the tool in a generic fire-and-forget shim →
  business tools don't import hitl_core. Also fixes #12-1 (`_session_id` arg
  pollution — only inject for hitl_mode tools) and absorbs #8.

#### HITL-P3 — nice-to-have
- #10 action_type enum/builder — being partially done in Phase 2B (delegate:
  type + builder); remaining callers can migrate incrementally.
- Cross-process SLA timer idempotency residual of #3 (process-restart half).

### Phase 3 — Cross-agent HITL passthrough
- **Effort**: 2-3 weeks
- **Status**: designed, not started
- **Scope**: new `hitl_core/cross_agent.py` (`CrossAgentHitlBridge`); A2A `INPUT_REQUIRED` state carries hitl_payload in metadata; entry agent renders peer's HITL card with "from dc-agent" tag; cross-agent correlation id for audit; timeout/cancel handling.
- **Prereq**: Phase 2B stable for at least a week.
- **Complexity note**: failure modes (network flake / peer crash / operator non-response / concurrent HITLs) are the bulk of the work — needs 5-8 integration test scenarios.

---

## Known bugs / tech debt to revisit

### Skill feedback path was dead for the project's entire life
- **Status**: FIXED 2026-05 (`SkillEvolver._parse_json_response` was never implemented).
- **Follow-up**: now that `apply_feedback` + `evaluate_skill_creation` actually run, watch production for unexpected side effects (e.g. LLM rewriting skills badly). The A/B safety net (compliance bench gate) should catch regressions, but monitor the `SkillEvolver: rollback` log lines.
- **Action**: After 2 weeks of production data, decide whether the auto-skill-evolution is net-positive or should be gated behind manual approval. **Keep this item until that decision is made.**

### Legacy `hitl/*` LangGraph backend retired but referenced
- **Status**: open
- **Detail**: `HITL_BACKEND != "core"` raises `NotImplementedError`. The `hitl/` package is a thin schema stub; the implementation modules were never packaged. Some lifespan code still has `if _services.get("hitl_router")` branches for the legacy path.
- **Action**: Either fully remove the legacy branches + the `HITL_BACKEND` env knob, or restore the langgraph backend. Pick one; don't leave the half-state.

### `context_budget_v2.py` co-exists with `context_budget.py`
- **Status**: open (gradual migration)
- **Detail**: v2 is the new priority-budget algorithm; some callers still use v1.
- **Action**: Finish migrating all callers to v2, then delete v1.

### `loop.py` is too large (~3000 lines)
- **Status**: open
- **Action**: Split into turn-loop / tool-dispatch / verify / stop-check modules sharing a `_LoopContext` dataclass.

### StopPolicy doesn't know about user cancel
- **Status**: open
- **Detail**: The frontend "stop" button isn't wired to the loop.
- **Action**: Add `StopOutcome.USER_CANCELLED` + wire the WebUI stop button through to the runtime loop.

---

## Done (kept briefly for reference, delete after a release cycle)

- ✅ **C1 — Prometheus `/metrics` endpoint** (2026-05) — `prometheus_client` text format at `/metrics`; counters for LLM calls / tool dispatch / HITL pending.
- ✅ **C2 — FastAPI / httpx auto-instrumentation** (2026-05) — `opentelemetry-instrumentation-fastapi` + `-httpx` wired in `runtime/tracing.py:configure()`; every HTTP request/outbound call auto-spanned when tracing enabled.
- ✅ **D1 — LLM call semaphore** (2026-05) — `cfg.llm.max_concurrent_calls` (default 4); `OllamaEngine` gates `_chat_impl` through an `asyncio.Semaphore` so a single query's 20+ internal LLM calls can't saturate Ollama.
- ✅ **Phase 2A — Business profile layer** (2026-05) — `profiles/{default,lan,dc}/` decouples domain-specific tools/skills from the common framework. `AGENT_PROFILE` env (or `agent.profile` config) selects active profile. `audit_profiles.py` enforces (a) callable/metadata alignment, (b) cross-profile tool isolation, (c) `default` is empty (decoupling proof), (d) no framework hard-imports of specific profile packages, (e) every loader call passes `profile=`. Details: `profiles/DESIGN.md`, `ARCHITECTURE.md §12`.
- ✅ **H2 — Async HITL (fire-and-forget)** (2026-05) — three-mode HITL design: H1/H3 (existing sync) + H2 new. `request_approval_async()` API; `_async_registry` module dict; turn-start inject queue → `state.confirmed_facts`; SSE soft-notify; SLA timeout via `on_resolved(decision=None)`; operator-approve triggers follow-up agent turn so LLM auto-uses the new fact; `query_radius_logs` demo tool lives in `profiles/lan/`; FE 🔔 banner + follow-up answer card. MFA deferred. Details: `ARCHITECTURE.md §8.6`, `hitl_core/DESIGN.md §3.2.5`.
