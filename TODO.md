# TODO — Production Readiness & Roadmap Tracker

> Tracking doc for deferred work. **Rule: when an item is done, delete it from this file.**
> Items that remain are not yet done. Last updated: 2026-05 (v13 + tech-debt sweep: legacy-hitl removal, user-cancel, auto-evolve switch, priority budget wired; loop.py split fully done — all _stream_impl phases extracted).

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

## L0/L1 two-layer architecture — clean capability/business separation

**Goal**: formalise two layers so external teams can build any agent (network,
coding, Q&A, …) on a stable base.
- **L0 — pure capability framework**: independent modules (runtime loop, memory,
  retrieval, hitl_core, skills engine, a2a, task) wired by integration + config.
  Knows nothing about any business domain.
- **L1 — business framework**: a business designer expresses an actual problem
  by declaring tools/skills/flows in a profile; the L0 skeleton hooks them
  together. `profiles/` already IS the start of L1 (framework → profiles
  dependency arrow, `default` profile = 0 tools = decoupling proof).

**Why now**: the macro seam exists (profiles), but business concepts have leaked
into L0, and the `memory → skill → delegation → tool → HITL` linkage chain has
no clean per-seam extension point — so adding/altering a domain risks
cross-cutting breakage. Target: each seam in that chain is a documented,
pluggable hook so new developers extend by filling in clearly-bounded slots.

### Stage A — concept repatriation ✅ DONE (2026-05)
Moved hardcoded business nouns out of L0; behaviour unchanged (verified 7/7 audits + 202 tests).
- `DeviceRef` → generalised to neutral `ResourceRef` (id/label/`type`/meta; `type` defaults to `"resource"`). `DeviceRef` kept as a back-compat alias; `runtime/__init__` exports both. Internal annotations migrated to `ResourceRef`.
- `RuntimeConfig.editable_hitl_tools` hardcoded default (`edit_device_config`/`rollback_deploy`/`restart_service`) → EMPTY L0 default; now injected from `cfg.tools.editable_hitl_tools` (added to config.py loader + config.yaml + wired in webui/backend `RuntimeConfig(...)`). `hitl_tool_names` was already config-injected.
- Test: `tests/test_l0_l1_separation.py` (ResourceRef neutral + alias; editable_hitl_tools empty-default + injectable).
- **Surfaced for Stage B** (marked in code, not yet fixed): `integrations/adapters/hitl_executor.py` falls back to `build_default_device_coreferencer()` when none injected — an L1 network default leaking into an L0 adapter. The `Coreferencer` class itself is already domain-neutral/parameterised; only the default needs to move to profile injection.

### Stage B — business logic externalised onto the linkage chain ✅ DONE (2026-05)
Made the tool→HITL seam pluggable; network logic moved to L1 (verified 7/7 audits incl. module_independence + 208 tests).
- **Batch HITL resolver**: the ~165-line network "should this destructive tool call expand into a multi-target batch?" logic (Path A same-name dedup + Path B device-prose fabrication, with device-id regex + `device_id`/`device`/`target` field mutation) was extracted from `_handle_tools` into `profiles/network_batch_resolver.py`. The L0 loop now exposes an injected `batch_resolver_fn` (precedent: `delegate_fn`); contract `(tool_name, tool_args, llm_response, hitl_tool_names, confirmed_facts, all_parsed) -> Optional[list[(name,args)]]`. No resolver → single-target HITL (domain-free default). Profile picks it via `profiles.get_batch_resolver_for_profile(profile)` (lan/dc → network resolver, default → None), wired in webui/backend.
- **Coreferencer default**: `HitlExecutor` defaulted to `build_default_device_coreferencer()` (network default leaking into an L0 adapter). Added `build_neutral_coreferencer()` (empty patterns → always "no entity") as the L0 default; main.py injects the device coreferencer only for lan/dc profiles.
- `runtime/loop.py` does NOT import `profiles/` (resolver injected) — `audit_module_independence` stays green.
- Test: `tests/test_batch_resolver.py` (profile selection; loop stores injected fn; Path A / Path B / single-device-no-batch).
- **Note**: the `memory→skill→delegation` seams were audited and found already clean (delegation uses injected `delegate_fn`; skill selection goes through the parameterised `SkillCatalogService`; memory recall is domain-neutral). The concrete L1 leak on the chain was concentrated at the tool→HITL seam, now resolved.

### Stage C — declarative L1 business-flow layer (large, DEFERRED)
- **Effort**: 1-2 weeks. **Status**: deferred — vision clear, requirements not
  yet mature (need 2-3 real domains before abstracting the flow model, else we
  abstract wrong).
- **Scope**: profiles declare not just capabilities but *flow orchestration*
  (e.g. a network-triage SOP: check status → on anomaly check neighbours → …);
  L0 provides a flow-engine that executes the declared graph. Upgrades profiles
  from "inject capabilities" to "inject flows". Revisit after Stage B ships and
  a second non-network domain is built on the framework.

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
- **Status**: **P3-a + P3-b DONE (2026-05)** — mode B (delegated-local HITL +
  active resume) is end-to-end. P3-c (correlation audit chain) + P3-d
  (passthrough mode C + failure-mode hardening + persistence) remain.
- **Modes**: A=Local (✅ pre-existing), B=Delegated-local — peer's operator
  approves, result flows back and the originator auto-resumes (✅ this round),
  C=Passthrough — peer's card surfaced back to the entry agent for approval
  there (❌ P3-d), D=Chained — one request, two HITLs across agents correlated
  (partial — correlation_id plumbed, audit join in P3-c). Default = B.
- **P3-a (done)**: `TaskState.AWAITING_PEER_HITL`; dispatcher marks the outbound
  task with it + stamps `peer_agent` / `peer_interrupt_id`; lan-side read-only
  "⏳ awaiting <peer> approval" badge in the Delegations tab (approval happens on
  the peer's console, NOT locally).
- **P3-b (done — the closed loop)**: `_unwrap_a2a_event` now translates the
  peer's `input-required` A2A status (message=interrupt_id) into a
  `hitl_interrupt` chunk carrying that id — **this closed the gap where the
  interrupt_id was silently dropped, so the correlation could never form**.
  `CrossAgentHitlBridge` (`task/inter/cross_agent_hitl.py`) correlates
  `(peer_agent, interrupt_id)` ↔ the local awaiting session, and on the
  delegated-to side maps `interrupt_id` → source-agent callback info. New A2A
  endpoint `POST /api/v1/a2a/hitl_resolved` on the originator. On peer approval,
  `HitlExecutor._tool_call_resumer` → `_maybe_callback_source_agent` resolves the
  source agent's URL via the injected peer registry and POSTs the result back;
  the originator's `handle_cross_agent_resume` injects it (H2 `enqueue_async_inject`
  reuse) and **actively drives one synthesis turn (A2)**, buffering the answer
  for the frontend `/chat/resumptions` poll (the original SSE has closed). Tests:
  `test_cross_agent_hitl.py` (unwrap translation, full correlation chain,
  double-resume guard, buffer-on-no-driver).
- **P3-b hardening — delegation storm elimination + reliable UI delivery (done, 2026-05)**:
  live two-agent runs exposed that mode B "worked" but spawned 4-5 duplicate DC
  inbound tasks per request, returned contradictory diagnoses, and never showed
  the final answer in the UI. Four root causes, four fixes, all on the LAN
  (originator) side + frontend:
  - **Single delegation gate (replaces 3 fragile env_ctx guards)**: the old
    per-target count / pending-set / resume-flag guards lived on `env_ctx`,
    which is per-`execute_query` and reset every time the resume driver started
    a fresh synthesis turn → duplicates slipped through. Replaced with ONE gate
    in `task/delegation.py build_delegate_fn` keyed on TaskStore state: identity
    = `(session_id, target_agent)`; if an outbound `scope==INTER` task to that
    peer is non-terminal (RUNNING / AWAITING_PEER_HITL / PENDING — terminal =
    COMPLETED/FAILED/CANCELLED), suppress (no task, no dispatch). TaskStore is
    durable across turns AND streams and is the same store the UI reads, so the
    gate and UI can never disagree. Test `test_delegation_gate.py` (6 cases).
  - **Park on case2 peer HITL**: when the peer raises an operator-approval HITL,
    the originating stream must WAIT for the async stage-2 result, not busy-loop
    (busy-looping races the result callback: the instant it flips the task to
    COMPLETED the gate releases and the next turn re-delegates). The loop now
    emits a `cross_agent_parked` marker + a deterministic interim and `return`s
    (ends the stream); the async result callback drives the synthesis turn.
    Test `test_delegation_park_on_peer_hitl.py`.
  - **Per-request re-delegation block + synthesis-turn hard-block**: a case1
    (synchronous, no-HITL) peer return that the LLM reads as inconclusive (e.g.
    DC asks "shall I grant?" instead of acting) was re-delegated to the same
    peer — the gate can't catch it (the task is already terminal). A per-stream
    `_delegated_targets_this_request` set suppresses re-delegating the same peer
    within one request, forcing the LLM to synthesize/degrade (spec point 2).
    Separately, a `_cross_agent_resume` per-turn flag hard-blocks any DELEGATE
    in a synthesis turn. Test `test_delegation_park_on_peer_hitl.py::TestCase1NoReDelegate`.
  - **Directive leak fix**: `[DELEGATE:]` / `[SKILL_LOAD:]` are now stripped from
    the user-visible token stream (previously only `[TOOL:]` was), so a
    pure-directive response (e.g. a suppressed re-delegation) doesn't leak raw
    directive text. `strip_delegate_directives` in `runtime/directive_parser.py`.
  - **Frontend resumption delivery**: the original SSE closes on park, so the
    async synthesis answer is buffered in `_pending_resumptions` for the
    `/chat/resumptions` poll. Two frontend bugs fixed: (a) the dedup key must
    include `phase` — approval (interim) and result (final) share one
    `correlation_id`, so keying on it alone dropped the final answer as a
    "duplicate" of the interim; (b) the poll must only stop on a terminal
    `phase==result` item, not on the first item (the approval interim).
    `webui/index.html startResumptionPoll`, gated on a `cross_agent_parked`
    marker so normal queries don't poll.
- **Remaining — P3-c (med)**: thread `correlation_id` into the cross-agent audit
  log so a request that chains lan-radius-HITL + dc-app-HITL (mode D) shows as
  one joined trail.
- **Remaining — DC-side robustness (low/product)**: DC's LLM sometimes diagnoses
  a permission gap then *asks* "shall I grant?" (case1) instead of firing the
  grant HITL (case2). The framework now handles both, but steering DC to act
  directly is a `dc_app_access_diagnose` skill-prompt tweak, not a framework fix.
- **Remaining — P3-d (high)**: passthrough mode C; failure-mode hardening
  (peer crash / operator non-response / callback POST failure → SLA watchdog
  re-surface; lan session expired on callback); persist `CrossAgentHitlBridge`
  + resumption buffer (today in-memory, lost on restart); auth/signing on the
  `/hitl_resolved` endpoint (today basic validation: must match an awaiting
  record); 5-8 live integration scenarios.

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

### `loop.py` is too large (~3300 lines)  ✅ DONE (2026-05)
- **Result**: loop.py 3428 → ~3250; `_stream_impl` 1448 → ~875. Extracted, each step verified (7/7 audits incl. module_independence + full suite):
  - `runtime/loop_helpers.py` — pure stateless helpers (`strip_thinking`, `is_complete`, `skill_loads_in`, `format_final`, `query_mentions_concrete_target`, `call_key`, `build_tool_ledger`, `page_default_size_for_ledger`). loop keeps same-named wrappers/aliases.
  - `runtime/loop_types.py` — public types (`QueryComplexity`, `DelegationMode`, `ForkContextPolicy`, `VerificationResult`, `ComplexityDecision`, `RuntimeConfig`, `LoopResult`); re-imported so `from runtime.loop import RuntimeConfig` is unchanged.
  - `runtime/loop_context.py` — `_LoopContext` per-turn mutable state, passed `(self, ctx)` to phase methods.
  - `_stream_impl` phases extracted: `_refresh_recall`, `_refresh_skills`, `_assemble_context` (unified legacy/priority — also fixed priority never applying in the streaming path), `_run_clarification_gate` (async-gen + `_clarification_terminal` sentinel), `_handle_tools` (async-gen + `_tools_terminal` sentinel; HITL stop semantics preserved).
  - New tests: `test_clarification_gate.py`, `test_handle_tools_phase.py` (incl. "destructive tool does not execute without approval" HITL safety assertion).
  - Docs refreshed: `runtime/DESIGN.md` §4.7/§4.8 + file table + §7; `ARCHITECTURE.md` tech-debt + sprint status.
- **Intentionally left in `_stream_impl`**: token streaming, DELEGATE handling, completion/stop-check + ledger writes — tightly coupled to loop control flow, not worth extracting.

---


### Cross-domain tool misinvocation guidance (Phase 3 follow-up, B — deferred)
- **Context**: the shared HITL watch-list once caused lan to raise a phantom HITL card for `dc_grant_app_access` (a dc-only tool). Fixed (option A) by gating watch-list HITL on the tool being in THIS agent's local registry — see `runtime/loop.py` CAP5 gate (`_needs_hitl = name in hitl_tool_names and name in ctx.tool_reg`) + same-name batch detection guard. Regression: `test_handle_tools_phase.py::test_watchlisted_tool_not_in_local_registry_does_not_raise_hitl`.
- **Remaining (B)**: when the LLM names a tool that is NOT local but a known peer HAS it, proactively steer the LLM to `[DELEGATE:<peer>]` instead of letting it fall through to a plain "tool not registered" error. Today (post-A) a misinvoked cross-domain tool just errors and the LLM usually self-corrects to delegation on the next turn; B would make that explicit (e.g. inject guidance "tool X belongs to peer Y — use DELEGATE"). Needs peer-capability lookup in the tool-miss path. Do only if misinvocation proves frequent in practice.

## Done (kept briefly for reference, delete after a release cycle)

- ✅ **Cross-agent resume callback no longer blocks on the peer's LLM turn** (2026-05, A2A Phase 3 mode B) — root-caused from real two-agent logs: the whole resume chain was synchronous — dc resumer `await POST(lan /hitl_resolved, httpx timeout)` → lan handler `await handle_cross_agent_resume` → `await _drive_cross_agent_resume` → `await executor.execute_query(...)` (a full 30-60s LLM synthesis turn on qwen3.5:27b). The 15s POST timeout fired long before lan's turn finished, so dc's resumer hung mid-approval (never reached tool-exec/_complete_inbound_by_interrupt → dc inbound task stuck; lan's turn result orphaned; lan never resumed). Logs proved bridge inbound record + lan awaiting record were both created with matching interrupt_id (f4192096-98d) — ruling out id/agent mismatch; the break was purely the synchronous blocking callback. Fix (option A): `webui/backend.py handle_cross_agent_resume` now SCHEDULES the resume turn via asyncio.create_task (held in module-level _resume_tasks set so it isn't GC'd) and returns True immediately, so the peer's POST gets a fast 200; the completed answer still reaches the UI via the existing _pending_resumptions buffer + /chat/resumptions poll + live SSE. Backstop: dc-side callback httpx timeout 15s→30s. Test: `test_cross_agent_hitl.py::TestResumeDriverContract::test_handle_cross_agent_resume_does_not_block_on_slow_driver` (fastapi-guarded skip). NOTE: real two-agent verification still pending — confirm dc inbound flips DONE right after approval, dc callback POST logs →200 (not timeout), and lan's resume turn streams the final answer into its UI.

- ✅ **Inbound-delegation-stuck-PENDING fix** (2026-05) — when the dc agent served a peer [DELEGATE:] request whose tool required HITL, execute() returned at the interrupt and parked the inbound TaskDefinition in PENDING (metadata['awaiting_hitl_id']=interrupt_id). The operator approves LATER via _tool_call_resumer, which runs the tool + calls back the originator but NEVER re-reached execute()'s completion path — so the inbound task stayed PENDING forever, the delegating agent's view never closed, and (because the delegation never completed) lan kept RE-DELEGATING ("请再次...") piling up duplicate inbound tasks on dc. Fixed with `HitlExecutor._complete_inbound_by_interrupt(interrupt_id, decision, result_text)` (scans task_store.list_pending() for the task whose awaiting_hitl_id matches; approve→COMPLETED+result, reject→FAILED+error, pops awaiting_hitl_id, stamps completed_at), wired into _tool_call_resumer at BOTH approve + reject sites alongside _maybe_callback_source_agent. Tests: `test_inbound_delegation_completion.py` (approve/reject/unknown-interrupt no-op/no-store-safe/only-matching-completes). Fixing the PENDING closure also removes the duplicate-re-delegation source. NOTE: real two-agent verification still pending — confirm after dc HITL approval the dc inbound task flips PENDING→DONE and lan stops re-delegating.

- ✅ **Phantom cross-profile HITL fix** (2026-05) — the shared `hitl_tool_names` watch-list caused an agent to raise a HITL approval card for a tool it doesn't have (lan popping a card for dc-only `dc_grant_app_access`), which then failed in the resumer with "tool not registered" AND produced a duplicate card racing the delegated-to agent's real one (two-sided approval, dc task stuck pending, lan resume 404). Fixed in `runtime/loop.py`: watch-list HITL now requires the tool to be in THIS agent's local registry (`_needs_hitl = name in hitl_tool_names and name in ctx.tool_reg`), plus the same guard on same-name destructive batch detection. Confirmed via real two-agent run that the skill-routing (A+C) fix worked: dc correctly diagnosed alice's missing CRM role via dc_get_app_acl, and dc-side HITL on dc_grant_app_access fired correctly. Tests: `test_handle_tools_phase.py` (phantom-HITL case + local-tool-still-gates guard). Follow-up B (steer misinvoked cross-domain tool to DELEGATE) deferred.

- ✅ **Cross-agent fault-scenario mock tools** (2026-05) — added the user/app access-control tools needed to mock "user alice cannot access app CRM" end-to-end across lan+dc. LAN (`profiles/lan`): `list_users`, `get_user_access` (RADIUS/802.1X/NAC/VLAN admission, read-only), `check_nac_policy`, `grant_user_access`/`revoke_user_access` (HITL). DC (`profiles/dc`): `dc_list_apps`, `dc_get_app_acl`, `dc_check_user_app_access` (read-only diagnostic), `dc_grant_app_access`/`dc_revoke_app_access` (HITL). Method 甲: queries read-only, only grant/revoke HITL-gated (dc grant fires the DC-side HITL = Phase 3 mode B). Datasets are consistent: alice is admitted on LAN but holds no CRM role on DC (the root cause). Watch-list + editable_hitl_tools updated; HITL safety-net warning softened for the now cross-profile watch-list; fixed a latent bug in audit_profiles check-5 (relative `tests/` path wasn't exempted). Test `test_access_scenario.py`. 223 tests pass.

- ✅ **A2A Phase 3 P3-b hardening — delegation storm fix + reliable UI delivery** (2026-05) — mode B was validated end-to-end but live runs spawned 4-5 duplicate DC inbound tasks per request, returned contradictory diagnoses (DC re-diagnosing already-mutated state), and never rendered the final answer in the UI. Four root causes fixed: (1) **single TaskStore-state delegation gate** in `task/delegation.py` (identity `(session, target)`, suppress non-terminal `scope==INTER` task) replacing three fragile per-`execute_query` env_ctx guards that reset on every resume turn; (2) **park-on-peer-HITL** — the originating stream now ends with a `cross_agent_parked` marker + interim and waits for the async result callback instead of busy-looping (which raced the callback and re-delegated the instant the task went terminal); (3) **per-request re-delegation block** (`_delegated_targets_this_request`) + **synthesis-turn hard-block** (`_cross_agent_resume`) to stop case1 (no-HITL) re-delegation the gate can't catch; (4) **frontend `/chat/resumptions` poll** dedup-by-`phase` + stop-only-on-terminal fix (approval interim and result final share one `correlation_id`, so the final answer was being dropped as a duplicate and the poll stopped after the interim). Also: `[DELEGATE:]`/`[SKILL_LOAD:]` now stripped from the visible token stream (`strip_delegate_directives`). Tests `test_delegation_gate.py` (6), `test_delegation_park_on_peer_hitl.py` (park + case1-no-re-delegate). 255 tests pass.
- ✅ **A2A Phase 3 P3-a + P3-b — cross-agent HITL mode B** (2026-05) — lan delegates to dc, dc raises a HITL approved on dc's console, result flows back and lan auto-resumes via an active synthesis turn. Closed the core gap: `_unwrap_a2a_event` was silently dropping the peer's `input-required` status so the interrupt_id never reached the originator — now translated into a `hitl_interrupt` chunk. New: `TaskState.AWAITING_PEER_HITL`, `CrossAgentHitlBridge`, A2A `/hitl_resolved` callback endpoint, `HitlExecutor._maybe_callback_source_agent` (registry-resolved POST back), webui active-resume driver + `/chat/resumptions` poll, lan read-only "awaiting peer approval" badge. Tests `test_cross_agent_hitl.py`. P3-c (audit chain) + P3-d (passthrough mode C + failure hardening + persistence + auth) remain. 215 tests pass.

- ✅ **H2 async-HITL live bugs** (2026-05, from real lan-agent run) — two runtime-path bugs the static tests missed: (1) `_submit_hitl_decision` H2 follow-up raised `NameError: _message_history` — it is a module-level handler and cannot see create_webui_app's closure local; fixed by publishing `_message_history` into `services` and reading it from there (+ static scope-leak regression test `test_hitl_submit_scope.py`). (2) `query_radius_logs` demo autoreply `NoneType.deliver` — the lan tool resolved `services["hitl_router"]` (a stub-None key since the Item-2 legacy-hitl cleanup) instead of the real `hitl_core_router`; fixed to resolve hitl_core_* first (router + audit). Both verified; full suite 210 passed.

- ✅ **L0/L1 separation — Stage B** (2026-05) — tool→HITL seam made pluggable: the ~165-line network batch-HITL logic moved from `_handle_tools` to `profiles/network_batch_resolver.py`, injected via `AgentRuntimeLoop.batch_resolver_fn` (None→single-target HITL). Coreferencer default made neutral in L0 (`build_neutral_coreferencer`), device one injected per-profile. `runtime/loop.py` imports no `profiles/` — module_independence green. Tests `test_batch_resolver.py`. memory/skill/delegation seams audited = already clean. Stage C (declarative flow layer) still deferred.

- ✅ **L0/L1 separation — Stage A** (2026-05) — concept repatriation: `DeviceRef`→neutral `ResourceRef` (back-compat alias kept); `RuntimeConfig.editable_hitl_tools` business default emptied + injected from `cfg.tools.editable_hitl_tools` (config.py/config.yaml/backend wired). L0 runtime now carries no hardcoded network tool names. Test `test_l0_l1_separation.py`. Stage B (hook-based chain externalisation) is next; Stage C (declarative flow layer) deferred. See "L0/L1 two-layer architecture".

- ✅ **loop.py decomposition COMPLETE** (2026-05) — finished step 4e: extracted `_handle_tools` (per-turn tool dispatch + HITL gate + execution + post-verify, async-gen with `_tools_terminal` sentinel; HITL stop semantics preserved + tested). loop.py 3428→~3250, `_stream_impl` 1448→~875. Full module set: `loop_helpers`/`loop_types`/`loop_context` + phase methods. New tests `test_clarification_gate.py` + `test_handle_tools_phase.py`. Docs refreshed (`runtime/DESIGN.md` §4.7-4.8, `ARCHITECTURE.md`).

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
