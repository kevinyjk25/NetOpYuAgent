# TODO — Production Readiness & Roadmap Tracker

> Tracking doc for deferred work. **Rule: when an item is done, delete it from this file.**
> Items that remain are not yet done. Last updated: 2026-05 (v13 + tech-debt sweep: legacy-hitl removal, user-cancel, auto-evolve switch, priority budget wired; loop.py split: phase 1 + _stream_impl steps 4a-4d done, 4e remaining).

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

### Phase 2B — Capability-based delegation ✅ DONE (2026-05)
Delivered + validated end-to-end with two live agents (LAN :8000 ↔ DC :8001,
real qwen3.5:27b). Decisions held: explicit `[DELEGATE:agent_id]` only (no
auto-delegate); default fresh (`#forked` opt-in shares parent confirmed_facts);
registry `_pick` for `*capability`, direct lookup for explicit id; entry agent
audits the delegation boundary only; `[DELEGATE:]` MUTUALLY EXCLUSIVE with
`[TOOL:]` in one turn (tool wins + nudge).

Shipped pieces: `[DELEGATE:target[#mode]]` parser + audit; `runtime/loop.py`
injected `delegate_fn` + `_handle_delegate` (fresh/forked, `source_agent`
tagging, result injected via `state.record_new_fact()` so it survives the
per-turn `budget.assemble()` rebuild); `task/delegation.py` `build_delegate_fn`
factory (resolve -> TaskDefinition -> dispatcher stream -> graceful degrade);
`main.py`/`webui/backend.py` wiring; peer-aware prompt section; WebUI delegate
badge. Plus debt #10 (`ProposedAction` builders + `ActionTypePrefix`) and debt
#7/#12-3 (`build_resumption_query`). Tests: `test_delegate_directive`,
`test_delegation_wiring`, `test_delegation_e2e`, `test_delegation_a2a_unwrap`.
See ARCHITECTURE.md §8.7, PHASE_2B_DESIGN.md. *(Kept until a release cycle.)*

**Note on the early design notes (now corrected):** the "A2ATaskDispatcher URL
bug" was a misread — the server does expose `POST /stream` and the dispatcher
was correct. `build_task_services()` was likewise already wired (as
`create_task_system`). The real transport bugs surfaced only under live
two-agent load and were fixed in the transport hardening pass below.

#### Phase 2B transport hardening ✅ DONE (2026-05, v8->v13)
Five bugs found only in live two-agent runs (each has a regression test):
- **A2A event-envelope unwrap** — the peer streams A2A protocol events
  (`TaskArtifactUpdateEvent` etc.) with token text nested at
  `artifact.parts[].data.token`; the dispatcher forwarded the raw envelope so
  the parent loop saw no top-level `token` and accumulated nothing. Fix:
  `A2ATaskDispatcher._unwrap_a2a_event` (`test_delegation_a2a_unwrap`).
- **EventQueue never finalised -> ReadTimeout** — `HitlExecutor.execute` (DC's
  A2A inbound entry) called non-existent `enqueue_event_status/_message`
  helpers; every enqueue silently failed and the queue was never sealed, so
  the consumer blocked forever and the parent hit a ReadTimeout. Fix: rewrote
  to use real `enqueue_event(<A2AEvent>)` + guaranteed `MessageEvent` finalize
  on every exit path.
- **300s SSE stall during peer's slow first token** — the peer runs a full
  agent loop (classify + 2-3 LLM turns, minutes on local qwen3.5) before its
  first token; the parent's `sse_stall_timeout` cancelled first. Fix:
  `_with_heartbeat` injects a no-op `node_step` keep-alive every 30s of
  silence; non-terminal peer status events also map to a brief progress chunk
  (`test_dispatcher_heartbeat`).
- **Double-counted final answer** — DC emitted `final_text` both as streamed
  tokens AND inside the sealing `MessageEvent`, so the parent LLM saw the
  analysis 2-3x and repeated itself. Fix: when tokens streamed, seal with the
  generic "Task completed." marker that the parent's unwrap filters out
  (`test_delegation_no_double_count`).
- **Outbound task stuck in RUNNING + raw-JSON render** — the parent's outbound
  `TaskDefinition` never transitioned past RUNNING (Delegations tab stuck) and
  the result rendered as `{"text": ...}` JSON. Fix: dispatcher updates state to
  COMPLETED/FAILED/PENDING on stream end + persists; detail panel renders the
  markdown answer (`test_delegation_outbound_state`).

**Follow-up — heartbeat interval not configurable**
- **Effort**: 15 min
- **Why**: `heartbeat_s = 30.0` is hardcoded in `A2ATaskDispatcher`. It must
  stay comfortably under `sse_stall_timeout_seconds` (default 180s); fine
  today, but an operator who lowers the stall timeout could starve it.
- **Action**: read `heartbeat_s` from config (e.g. `task.delegation_heartbeat_s`,
  default 30) and assert `heartbeat_s < sse_stall_timeout_seconds` at startup.

**Still NOT in 2B** (future): cross-agent HITL passthrough (Phase 3 — but the
provenance plumbing is now in place, see below), multi-hop, auto-delegate,
parallel fan-out, cross-agent memory writeback.

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
- **Effort**: #4 follow-up UI loading 1.5d; #7 resumption-query ✅ DONE
  (`build_resumption_query` in `runtime/loop.py`, shared by the H2 follow-up
  turn and delegation result injection).
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
- #10 action_type enum/builder — ✅ DONE in Phase 2B (`ActionTypePrefix` +
  `ProposedAction.tool_call/batch/diagnostic/delegate` builders). Existing
  hand-rolled `action_type` strings can still migrate to the builders
  incrementally, but the contract exists.
- Cross-process SLA timer idempotency residual of #3 (process-restart half).

### Phase 3 — Cross-agent HITL passthrough
- **Effort**: 2-3 weeks (reduced — provenance plumbing now done, see below)
- **Status**: designed, not started; **provenance groundwork shipped in v13**
- **Already done (de-risks Phase 3)**: HITL provenance now flows end-to-end
  when a delegated peer raises a card — `task/delegation.py` stamps
  `source_agent` / `source_session_id` / `source_query` into
  `TaskDefinition.metadata`; `A2ATaskDispatcher.dispatch` packs it into the A2A
  request `params.metadata`; `HitlExecutor.execute` extracts it into
  `env_context`; `_raise_tool_hitl` / `_raise_tool_hitl_batch` /
  `_raise_multi_mode` stamp the `HitlPayload` via
  `_extract_delegation_provenance`. `HitlPayload` carries `source_agent` /
  `source_session_id` fields (`test_delegation_provenance`). The peer operator's
  card can already show "Delegated from <source_agent>".
- **Remaining scope**: the *passthrough* itself — surface the peer's pending
  card back on the ENTRY agent's UI (today the peer's own operator must
  approve), via A2A `INPUT_REQUIRED` carrying the payload + a
  `CrossAgentHitlBridge`; cross-agent correlation id for the joined audit;
  timeout/cancel handling across the hop.
- **Prereq**: Phase 2B stable for at least a week (now is).
- **Complexity note**: failure modes (network flake / peer crash / operator
  non-response / concurrent HITLs) are the bulk of the work — needs 5-8
  integration test scenarios.

---

## Known bugs / tech debt to revisit

### Skill feedback path was dead for the project's entire life
- **Status**: FIXED 2026-05 (`SkillEvolver._parse_json_response` implemented) + **auto-evolve master switch added 2026-05**.
- **What shipped**: `cfg.skill_orchestration.auto_evolve_apply` (env `SKILL_AUTO_EVOLVE_APPLY`, default `true`). When `false`, the self-improvement loop still runs — computes feedback patches + new-skill proposals, records them in version history + logs — but does NOT mutate the live catalog (gated in both `apply_feedback` and `_register_skill`). Lets production run in "observe" mode. Test: `tests/test_skill_evolve_suggest_only.py`.
- **Follow-up (product decision, not code)**: after observing production, decide whether to default `auto_evolve_apply` to `false`. The A/B compliance bench still gates auto-applied patches when on. Monitor `SkillEvolver: rollback` / `suggested (suggest-only mode)` log lines.

### Legacy `hitl/*` LangGraph backend retired but referenced ✅ DONE (2026-05)
- **What shipped**: `HITL_BACKEND` now defaults to `core` (previously defaulted to `langgraph`, which crashed boot). All dead legacy branches removed from `main.py` (section-2 `else: raise`, section-5 defensive guard, section-6 legacy executor + `patch_hitl_graph` calls, the legacy `/hitl/*` mount block, the always-None `create_task_system` ternary). `patch_hitl_graph` function + its three exports deleted. A non-`core` value now fails fast with a clear "backend retired" message. **Bonus bug fixed**: the `/healthz`-style `pending_hitl` metric read the never-set `hitl_router` (always 0); now reads `hitl_core_router`.

### `context_budget_v2.py` co-exists with `context_budget.py` ✅ DONE (2026-05)
- **Correction to the old note**: v2 was never meant to *replace* v1 — its own docstring says it's an alternative selectable strategy. The real gap was that `cfg.context_budget.strategy = "priority"` wasn't wired into the loop.
- **What shipped (方案 A)**: `AgentRuntimeLoop` reads `cfg.context_budget.strategy` at construction; when `"priority"`, context assembly routes through `_assemble_priority` → v2 `TokenBudget` (reusing the legacy section formatters so rendered text is identical, but trimming low-priority sections — env, skills, retrieved memory — before high-priority ones — confirmed facts, recent tool results — under a hard `total_chars` cap). `"legacy"` (default) is unchanged. Degrades to legacy if v2 unavailable. Test: `tests/test_context_budget_priority.py`. Both modules now have a clear reason to exist; neither is deleted.

### StopPolicy doesn't know about user cancel ✅ DONE (2026-05)
- **What shipped**: `StopOutcome.USER_CANCELLED` added. WebUI Send button toggles to a red **Stop** button during streaming; clicking it aborts the fetch (`AbortController`). The backend SSE generator catches the resulting `CancelledError`/`GeneratorExit`, cancels the executor task (reclaiming the blocked Ollama request), marks the outcome `user_cancelled`, and preserves the partial answer in the session transcript (tagged "已取消 — 部分回答") while skipping durable memory writeback (a half-answer isn't a trustworthy fact to recall). The loop's existing `GeneratorExit`/`finally` SESSION_END cleanup makes cancellation safe.

### `loop.py` is too large (~3300 lines)  ← IN PROGRESS (2026-05)
- **Status**: structural extractions + first 4 `_stream_impl` phase extractions done (loop.py 3224 lines, down from 3428; `_stream_impl` 1334 lines, down from 1448). Only the largest phase (`_handle_tools`) remains.
- **Phase 1 — module-level extractions (verified: 7/7 audits incl. module_independence + 191 tests)**:
  - `runtime/loop_helpers.py` — pure stateless helpers (`strip_thinking`, `is_complete`, `skill_loads_in`, `format_final`, `query_mentions_concrete_target`, `call_key`, `build_tool_ledger`, `page_default_size_for_ledger`). The loop keeps same-named thin wrappers / aliases so all call sites are unchanged.
  - `runtime/loop_types.py` — all public type definitions (`QueryComplexity`, `DelegationMode`, `ForkContextPolicy`, `VerificationResult`, `ComplexityDecision`, `RuntimeConfig`, `LoopResult`). loop.py re-imports them and runtime/__init__ re-exports unchanged, so `from runtime.loop import RuntimeConfig` still works.
- **Phase 2 — `_stream_impl` decomposition steps 4a-4d (each verified: 7/7 audits + 194 tests)**:
  - `runtime/loop_context.py` — `_LoopContext` dataclass holds the per-turn mutable state (`state`, `tool_outputs`, `called_tools`, clarification flags, the memoised recall/skill caches + their refresh-cadence bookkeeping). Passed as `(self, ctx)` to each phase method instead of threading a dozen closure locals.
  - `_refresh_recall(ctx, ...)` + `_refresh_skills(ctx, ...)` — the two conditional memoisation phases, pure compute, mutate ctx caches.
  - `_assemble_context(...)` — unifies the legacy + priority context-assembly branch; now used by BOTH `run()` and `_stream_impl`. **Bonus fix**: the Tier-2 priority strategy was previously wired only into `run()`, so the streaming path (`_stream_impl`, what the WebUI uses) silently always used legacy — unifying here makes `strategy="priority"` actually apply in streaming.
  - `_run_clarification_gate(ctx, ...)` — the Type-#3 clarification gate (async generator; yields a `{"_clarification_terminal": True}` sentinel the caller returns on). New test: `tests/test_clarification_gate.py` (asks→sentinel / no-ask→empty / skip-after-turn-1).
- **Remaining — step 4e (deferred, highest regression risk, its own iteration)**: extract `_handle_tools` — the single-tool enforcement + HITL gate + tool execution + paginated-read nudge block (the largest remaining chunk of `_stream_impl`, an async generator that yields and can terminate the stream). Same rhythm: extract, full audit+test, keep `audit_module_independence` green.

---

## Done (kept briefly for reference, delete after a release cycle)

- ✅ **loop.py decomposition — phase 1 + _stream_impl 4a-4d** (2026-05) — extracted `runtime/loop_helpers.py` (pure helpers), `runtime/loop_types.py` (public types), `runtime/loop_context.py` (`_LoopContext` per-turn state); extracted `_stream_impl` phases `_refresh_recall`, `_refresh_skills`, `_assemble_context` (also fixed the priority strategy never applying in the streaming path), `_run_clarification_gate`. loop.py 3428→3224, `_stream_impl` 1448→1334. New test `test_clarification_gate.py`. Each step verified 7/7 audits + full suite. Only `_handle_tools` (step 4e) remains — see "Known bugs / tech debt".

- ✅ **Tech-debt sweep** (2026-05) — four of the five "Known bugs" items closed in one pass: (1) legacy LangGraph `hitl/*` backend fully removed, `HITL_BACKEND` defaults to `core`, plus a latent always-0 `pending_hitl` metric fixed; (2) `StopOutcome.USER_CANCELLED` + a real WebUI Stop button that aborts the stream, cancels the executor, and preserves the partial answer; (3) `auto_evolve_apply` master switch for the self-improving-skills loop (suggest-only vs apply); (4) `cfg.context_budget.strategy="priority"` wired into the loop via the v2 `TokenBudget`. New tests: `test_skill_evolve_suggest_only`, `test_context_budget_priority`. The only deferred item is the `loop.py` split (pure refactor). Details in "Known bugs / tech debt" above.

- ✅ **Phase 2B — Capability-based delegation + transport hardening** (2026-05) — explicit `[DELEGATE:agent_id|*capability[#forked]]` cross-agent task hand-off over A2A; validated live (LAN↔DC, qwen3.5:27b). Five transport bugs fixed under real load: A2A event-envelope unwrap, EventQueue finalize (ReadTimeout), 30s dispatcher heartbeat (300s stall), no-double-count of final answer, outbound task state transition + markdown render. HITL provenance (`source_agent`/`source_session_id`/`source_query`) now plumbed end-to-end (de-risks Phase 3). Details: ARCHITECTURE.md §8.7, PHASE_2B_DESIGN.md, and the Phase 2B section above.
- ✅ **Sprint-3-pre readiness** (2026-05) — `runtime/tracing.py` boots gracefully without OpenTelemetry installed; `HITLCheckpointConfig` defaults to sqlite (approvals survive restart) with env override; `SkillEvolver.set_bench_runner` A/B safety-net gate rejects compliance regressions. Pinned by `tests/test_sprint3_pre.py`.

- ✅ **C1 — Prometheus `/metrics` endpoint** (2026-05) — `prometheus_client` text format at `/metrics`; counters for LLM calls / tool dispatch / HITL pending.
- ✅ **C2 — FastAPI / httpx auto-instrumentation** (2026-05) — `opentelemetry-instrumentation-fastapi` + `-httpx` wired in `runtime/tracing.py:configure()`; every HTTP request/outbound call auto-spanned when tracing enabled.
- ✅ **D1 — LLM call semaphore** (2026-05) — `cfg.llm.max_concurrent_calls` (default 4); `OllamaEngine` gates `_chat_impl` through an `asyncio.Semaphore` so a single query's 20+ internal LLM calls can't saturate Ollama.
- ✅ **Phase 2A — Business profile layer** (2026-05) — `profiles/{default,lan,dc}/` decouples domain-specific tools/skills from the common framework. `AGENT_PROFILE` env (or `agent.profile` config) selects active profile. `audit_profiles.py` enforces (a) callable/metadata alignment, (b) cross-profile tool isolation, (c) `default` is empty (decoupling proof), (d) no framework hard-imports of specific profile packages, (e) every loader call passes `profile=`. Details: `profiles/DESIGN.md`, `ARCHITECTURE.md §12`.
- ✅ **H2 — Async HITL (fire-and-forget)** (2026-05) — three-mode HITL design: H1/H3 (existing sync) + H2 new. `request_approval_async()` API; `_async_registry` module dict; turn-start inject queue → `state.confirmed_facts`; SSE soft-notify; SLA timeout via `on_resolved(decision=None)`; operator-approve triggers follow-up agent turn so LLM auto-uses the new fact; `query_radius_logs` demo tool lives in `profiles/lan/`; FE 🔔 banner + follow-up answer card. MFA deferred. Details: `ARCHITECTURE.md §8.6`, `hitl_core/DESIGN.md §3.2.5`.
- ✅ **Peer-aware prompt** (2026-05) — `main.py` calls `llm.attach_peer_registry()` after retrieval wiring; `_build_peers_section()` from `LLMEngine` queries the registry at prompt-assembly time and injects an "AVAILABLE PEERS — delegate via [DELEGATE:agent_id]" section listing healthy peers + capabilities. Fixes the symptom where the LAN agent, asked about `spine-1 BGP EVPN`, exhausted local tools instead of delegating to dc-agent (because the prompt described the syntax but never said any peer existed). `audit_wiring.py` extended with a `REQUIRED_METHOD_CALLS` second pass so any future wiring-forgot regression fails CI. Details: `ARCHITECTURE.md §8.8`.
