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

### Phase 2 — Capability-based delegation
- **Effort**: 5-7 days
- **Status**: designed, not started
- **Scope**: `PolicyEngine.classify_and_route()` adds peer capability matching; `runtime/loop.py` gets a `_delegate_to_peer()` branch; reuse existing `A2ATaskDispatcher` (in `task/inter/coordinator.py` — never been exercised, expect bugs); chunks tagged with `source_agent` so UI shows "via wan-agent".
- **Prereq**: Phase 1 verified stable (peer discovery working in production).
- **Open product questions** (decide before building):
  1. Can agents cross-query domains (does LAN agent know which devices are WAN's)?
  2. Is `confirmed_facts` shared across agents (privacy boundary)?
  3. When multiple peers can handle a query, pick by capability score / load / round-robin?
  4. Does the entry agent's audit log include the delegated agent's execution detail (compliance)?

### Phase 3 — Cross-agent HITL passthrough
- **Effort**: 2-3 weeks
- **Status**: designed, not started
- **Scope**: new `hitl_core/cross_agent.py` (`CrossAgentHitlBridge`); A2A `INPUT_REQUIRED` state carries hitl_payload in metadata; entry agent renders peer's HITL card with "from wan-agent" tag; cross-agent correlation id for audit; timeout/cancel handling.
- **Prereq**: Phase 2 stable for at least a week.
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
