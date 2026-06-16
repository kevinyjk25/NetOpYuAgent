# TODO — Production Readiness & Roadmap Tracker

> **Rule**: when an item is done, move it to the DONE archive (bottom) in one condensed line.
> Everything in the OPEN section is *not yet done*. Last updated: 2026-06
> (capability-gap C-protocol shipped; run()=stream() collector; full design+code audit).

Priority legend (OPEN section):
- **P0** — blocks a safe production deploy. Do first.
- **P1** — important correctness / robustness / roadmap; do soon.
- **P2** — valuable but not urgent; schedule deliberately.
- **P3** — nice-to-have / opportunistic; do only if cheap or if pain proves real.

---

# ━━━━━━━━━━━━━━━  OPEN (not done)  ━━━━━━━━━━━━━━━

## P0 — Security / deploy blockers (before ANY internet-facing deploy)

### P0-1 — Auth-required-in-production startup check
- **Effort**: 0.5d
- **Why**: `cfg.auth.enabled=False` + `ENVIRONMENT=production` boots fine — an operator who forgets to flip auth ships an open API where anyone can approve destructive HITL.
- **Accept**: boot raises `RuntimeError` if `ENVIRONMENT=production` and `auth.enabled=false`, or if `auth.enabled=true` but the JWT secret env var is unset.

### P0-2 — CSRF protection on `/hitl/*` POST routes
- **Effort**: 1d
- **Why**: a logged-in operator visiting a malicious page could have it POST `/hitl/{id}/approve` and silently approve destructive ops.
- **Accept**: HITL approval endpoints reject requests without a valid CSRF token (min bar: require `X-Requested-With: XMLHttpRequest`).

### P0-3 — Secrets management
- **Effort**: 2-3d
- **Why**: device passwords live in env vars; `kubectl describe pod` exposes them.
- **Accept**: `cfg.secrets.backend` selector (env / vault / aws_sm); `secret_resolver.py` resolves `${...}` at startup; secrets never logged.

## P1 — Deployability + reliability + roadmap

### P1-1 — Dockerfile + docker-compose
- **Effort**: 1d. **Why**: no container artifacts; `uvicorn main:app` is a dev command. **Accept**: `Dockerfile` (non-root, healthcheck, gunicorn+uvicorn worker); `docker-compose.yml` with ollama + redis; `.env.production.example`.

### P1-2 — `/livez` + `/readyz` endpoints
- **Effort**: 1d. **Why**: `/health` returns 200 even when Ollama is dead, so k8s liveness can't detect a broken agent — requests hang 5 min. **Accept**: `/livez`=process alive; `/readyz`=Ollama reachable + memory DB writable + HITL store writable, 503 when any down.

### P1-3 — DEPLOYMENT.md operations runbook
- **Effort**: 1d. **Why**: all docs are dev-facing; an SRE has no sizing / upgrade / DR runbook. **Accept**: `docs/DEPLOYMENT.md` — external deps, min CPU/RAM/disk/GPU, upgrade flow (don't lose HITL pending state), alert thresholds.

### P1-4 — Backup cron + restore runbook
- **Effort**: 1d. **Why**: SQLite corrupt (power loss / disk full / volume delete) loses all facts + skill history + audit log. **Accept**: `scripts/backup.sh` (sqlite `.backup`, nightly, rotate 30d, optional S3); restore steps in DEPLOYMENT.md.

### P1-5 — Database migration framework
- **Effort**: 2d. **Why**: SQLite schema changes crash on old data or silently ignore it; no versioned migrations. **Accept**: Alembic or `migrations/` dir + `schema_version` table + serial apply at startup.

### P1-6 — Phase 6 P2: outcome-satisfied check → feedback (LAST evolution round)
- **Effort**: 3-5d. **Why**: closes the evolution loop. Judge whether a run satisfied the request; unsatisfied → demote/improve the skill; auto-optimize descriptions from journal dormant/never-loaded signals.
- **Bundle with P2 (do together — all touch loop outcome / journal)**:
  - **P3 first-use vs follow-up via loop state** — drop the append-marker heuristic in the P3 append-merger; track "skill loaded in a PRIOR request of this session" in loop state so a genuine follow-up is detected structurally (today: 也/还/再/顺便… markers, a known heuristic).
  - **HITL-gate-vs-LLM-self-pick tension** — the ambiguity HITL gate and the prompt's `[RELEVANT SKILLS]` candidate list coexist; an obvious match lets the LLM self-pick in turn-1 prose before the gate fires. Decide: hard-block before LLM, or accept soft "LLM-pick + preference-learning corrects".
  - **De-dupe `_evolve_cb`** — the `/evolution/sweep` endpoint and the backend P1 hook have a duplicated evolve-callback adapter; merge.

### P1-7 — HITL-P0: async-HITL state persistence (debt #1 + #2)
- **Effort**: 3-4d (declarative resumer handlers) + 2d (Redis SSE broker)
- **Why**: `_async_registry` / `_session_sse_emit` are per-process in-memory. Single-process is fine; `--workers N` or a restart loses pending async-HITL callbacks (operator approves → router finds no pending → silent drop). Delegation is NOT affected (persistent A2A task store). Needed for horizontal scale of the HITL feature.
- **Action**: serialize pending via a resumer-name handler registry (closures can't pickle — register declarative handlers like ResumeHandle does); SSE via Redis pub-sub or sticky sessions.
- **Interim (cheap)**: on startup, sweep store for PENDING+ASYNC_NONBLOCKING past SLA → mark EXPIRED + audit + UI "lost on restart" hint.

### P1-8 — A2A Phase 3 P3-d: passthrough mode C + failure hardening + persistence
- **Effort**: ~1-2wk. **Why**: remaining cross-agent-HITL robustness. **Action**: passthrough mode C (peer's card surfaced to the entry agent); failure modes (peer crash / operator non-response / callback POST failure → SLA watchdog re-surface; lan session expired on callback); persist `CrossAgentHitlBridge` + resumption buffer (today in-memory, lost on restart); auth/signing on `/hitl_resolved`; 5-8 live integration scenarios.

## P2 — Architecture / framework maturation

### P2-1 — Capability-gap detection: B (skill static dep-check) + A (plan-level)
- **Effort**: B ~1d (needs schema change first); A ~3-5d (planner destub).
- **Status**: **C done** (LLM declares `[CAPABILITY_GAP:]`, loop records + stops gracefully + journal ledger + UI). B and A remain — they catch what C (LLM self-report) misses.
- **B — A2A-aware static dep check**: validate a skill's `required_tools ⊆ (local registry ∪ delegable peer capabilities)` at SKILL_LOAD; if not, tell the operator at step 0. **PREREQ**: skill schema must split `required_tools` vs `optional_tools` first — `allowed_tools` today conflates hard deps, optional/branch tools, meta-tools, and cross-domain-delegated tools, so a naive subset check FALSE-POSITIVES on cross-domain skills (e.g. `app_access_troubleshoot` declares only LAN tools but completes via DC delegation). Must be a soft prompt-injected notice, never a hard block.
- **A — plan-level coverage**: for the long-chain case (request needs n tools, system has k<n), aggregate `route(query)` saturates — the k covered steps mask the n−k gap. Decompose first (WITHOUT showing the tool list, to avoid anchoring), then per-step CSI route + peer-AgentCard route → classify each step local / delegable / true-gap. Emit an upfront "step 4 of 4 lacks capability" report; if a coverable prefix contains destructive ops, HITL "still run the first 3 steps?" (partial-coverage execution is a human decision). This is the planner `_llm_decompose` destub. Dry-run (journal-only, no inject) first to calibrate capability_floor before enabling.
- **Limit**: A's detection quality = decomposition quality (bad granularity → false gaps). C stays the always-on safety net.

### P2-2 — HITL-P2 / Phase 2C: framework-ize H2 via PolicyEngine hitl_mode (debt #9 + #12-1 + #8)
- **Effort**: 3-4d (own phase). **Why**: `tool_meta` marks `hitl_mode: async_nonblocking` but PolicyEngine ignores it — H2 is hand-coded in `query_radius_logs`. **Ideal**: PolicyEngine reads the metadata → runtime wraps the tool in a generic fire-and-forget shim → business tools don't import hitl_core. Also fixes #12-1 (`_session_id` arg pollution — only inject for hitl_mode tools) + absorbs the demo-autoresponder off-switch (debt #8: `config.yaml agent.h2.demo_mode`, tool reads config not arg).

### P2-3 — HITL-P1: follow-up turn UX loading state (debt #4)
- **Effort**: 1.5d. **Why**: operator approve → follow-up turn blocks HTTP 10-60s with no loading state. **Action**: return `{async_followup_pending}` immediately + poll, or reopen SSE. (`build_resumption_query` part already done.)

### P2-4 — L0/L1 Stage C: declarative business-flow layer
- **Effort**: 1-2wk. **Status**: deferred — vision clear, requirements not mature (need 2-3 real domains before abstracting the flow model, else we abstract wrong). **Scope**: profiles declare flow orchestration (triage SOP: check status → on anomaly check neighbours → …); L0 provides a flow-engine executing the declared graph. Revisit after a second non-network domain exists.

## P3 — Nice-to-have / opportunistic

### P3-1 — A2A Phase 3 P3-c: correlation_id into cross-agent audit log
- **Effort**: ~1d. A request chaining lan-radius-HITL + dc-app-HITL (mode D) should show as one joined trail.

### P3-2 — Cross-domain tool misinvocation → steer to DELEGATE (Phase 3 follow-up B)
- **Effort**: ~1d. When the LLM names a non-local tool a known peer HAS, inject "tool X belongs to peer Y — use DELEGATE" instead of a plain "not registered" error. Today the LLM usually self-corrects next turn. Do only if frequent. (Overlaps P2-1's peer-capability lookup — share it.)

### P3-3 — Heartbeat interval not configurable
- **Effort**: 15min. `heartbeat_s=30.0` hardcoded in `A2ATaskDispatcher`; must stay under `sse_stall_timeout_seconds` (180s). **Action**: read from `task.delegation_heartbeat_s`, assert `< sse_stall_timeout_seconds` at startup.

### P3-4 — Skill journal persist path not auto-isolated
- **Effort**: 15min. `journal_persist_path` operator-set, off by default. If set to a literal path, two agents share one journal file. **Action**: document/route through `agent_data_dir()` with `${AGENT_ID}`.

### P3-5 — Pragmatic mode not profile-split
- **Effort**: 1-2d. `tools/pragmatic_tools.py` (real Netmiko/NAPALM) loads regardless of profile; a `dc` agent in pragmatic mode would still get LAN device tools. mock mode IS split. **Action**: split into `profiles/<id>/pragmatic_tools.py` when pragmatic multi-domain becomes real.

### P3-6 — Per-profile MCP / OpenAPI integrations
- **Effort**: 1d. Partially done (built-in netops MCP+OpenAPI now lan-only). **Remaining**: generalize the binary lan-or-not gate to a `Profile.integrations` declaration (MCP configs + OpenAPI specs per profile) so DC wires its own.

### P3-7 — Scheduler persistence + cron + push delivery
- **Effort**: varies. Prototype is in-memory (lost on restart), fixed-interval only, history-only (no push). Add sqlite/jsonl persistence + restart recovery, cron schedules, push delivery — when it leaves prototype scope.

### P3-8 — DC-side directness (product, not framework)
- DC's LLM sometimes diagnoses a permission gap then *asks* "shall I grant?" (case1) instead of firing the grant HITL (case2). Framework handles both; steering DC to act is a `dc_app_access_diagnose` skill-prompt tweak.

### P3-9 — auto_evolve_apply default (product decision)
- After observing production, decide whether to default `auto_evolve_apply` to `false`. A/B compliance bench still gates auto-applied patches when on. Monitor `SkillEvolver: rollback` / `suggested (suggest-only mode)` logs.

### Other still-not-in-scope (future)
- HITL-P3 residual: cross-process SLA timer idempotency (process-restart half).
- 2B future: multi-hop delegation, auto-delegate, parallel fan-out, cross-agent memory writeback.
- planner `_llm_decompose` STILL A STUB (the destub IS P2-1's A); `allowed_tools` runtime enforcement still pending.

---

# ━━━━━━━━━━━━━━━  DONE (condensed archive)  ━━━━━━━━━━━━━━━

> Delete entries after a release cycle. Newest first.

## Phase 6 — Agent evolution + capability-gap (2026-06)
- ✅ **C — capability-gap protocol**: LLM emits `[CAPABILITY_GAP: which step / what's missing]` when a required step has no fitting tool/skill and isn't delegable (iron rule in both system prompts). Loop parses (`directive_parser.find_capability_gap`), records a journal `capability_gap` event + structured `unresolved_point`, emits a `capability_gap` SSE chunk, stops gracefully, strips the marker from visible prose. `GET /webui/evolution/gaps` aggregates the "missing-capability ledger" (inverse of P1 solidify — demand signal for new tools) + 🕳 frontend button. Tests `test_capability_gap.py` (7). NOTE: C is LLM self-report (probabilistic); B+A (P2-1) are the structural complements.
- ✅ **run() = stream() collector** (JK chose option-a): ~300-line duplicated loop deleted, single execution path. **Fixed a real safety gap**: old run() had NO hitl_tool_names gate — non-streaming `/chat` could execute destructive tools without approval; now `stop_hitl` → `outcome=STOP_HITL` + `pending_interaction` card, tool NOT executed (test_run_wrapper.py). Non-interactive default suppresses ambiguity/clarification cards; approval HITL never suppressed. Connected dead `_clarification_resolved` flag. Terminal chunks gained outcome/tool_summaries/unresolved/turns (additive). H2 resume turn hard-blocks DELEGATE.
- ✅ **进化1 — skill choice → learn → auto**: weak-match ambiguity; choice → `skill_preference` fact (per user, ttl 90d); 3 stages learn(<0.5)/recommend(0.5-0.85,⭐pin)/auto(≥0.85); high-risk never auto; wrong auto → demote. Tests (11). **Config bug fixed**: new yaml keys dropped by the fixed-field dataclass — added all fields to dataclass+loader (lesson: new key = dataclass+loader+yaml).
- ✅ **P0 — real trajectory to evolver**: `extract_trajectory` reconstructs steps+tools+observations (was `solution_steps=[]`).
- ✅ **CSI v1** (`skills/capability_index.py`): unified interpretable similarity replacing 4 bespoke ones. Two-level clusters, hybrid sim with `reasons[]`, route/nearest_skill/cluster_trajectories, async embedder. Tests (9).
- ✅ **P1 — repeated trajectory → skill** (`trajectory_miner.py`): CSI-cluster journal trajectories, solidify recurrence ≥3, skip skill-covered. Lazy sweep every 5 complex tasks.
- ✅ **P3 — append → targeted merge** (`append_merger.py`): session-active = ground truth else CSI.nearest_skill, merge delta. Append-marker gated (heuristic → P1-6). Tests (6).
- ✅ **Manual endpoints**: `POST /evolution/sweep` + `GET /evolution/space` + Evolution frontend panel. `build_csi_from_profile` in webui layer (keeps capability_index.py decoupled). Harness `verify_csi_p1_p3.py`.
- ✅ **Full audit**: fixed 3 stale delegation-repeat tests; de-nested P0/P1/P3 hooks into sibling try-blocks; fixed harness embedder stub-vs-real misjudgment.

## Phase 5 — Anthropic skill format + scripts (2026-06)
- ✅ Full SKILL.md: folder+frontmatter, progressive disclosure, **script-as-tool** (AST-validated, registered as `<skill_id>__<script>`), references/assets, allowed-tools. Tests (11).

## Phase 4 — periodic task scheduler (2026-05, prototype)
- ✅ In-memory scheduler as agent tools (tool/query modes); `scheduler/` L0-pure; SCHEDULE tab. Tests (10). Remaining → P3-7.

## Multi-agent (Phase 2A / 2B / 3) — 2026-05
- ✅ **2A**: `profiles/{default,lan,dc}`, `AGENT_PROFILE`, profile-aware loaders, `audit_profiles.py`, tests (20); per-agent data isolation.
- ✅ **2B — capability-based delegation**: explicit `[DELEGATE:agent_id|*cap[#forked]]` over A2A, validated live. 5 transport bugs fixed under load (envelope unwrap, EventQueue finalize, heartbeat/stall, no-double-count, outbound state). HITL provenance plumbed.
- ✅ **Phase 3 P3-a+P3-b — cross-agent HITL mode B**: delegate → peer HITL → result flows back → auto-resume. `_unwrap_a2a_event` translates peer `input-required` → `hitl_interrupt`. `CrossAgentHitlBridge`, `/hitl_resolved`, active-resume + `/chat/resumptions` poll. Remaining → P1-8, P3-1.
- ✅ **P3-b hardening — delegation storm fix**: single TaskStore-state gate; park-on-peer-HITL; per-request re-delegation block + synthesis hard-block; frontend dedup-by-phase; async resume callback; inbound-stuck-PENDING fix; phantom cross-profile HITL fix.
- ✅ **Peer-aware prompt**: AVAILABLE PEERS section + `audit_wiring.py` REQUIRED_METHOD_CALLS.

## L0/L1 separation — 2026-05
- ✅ **Stage A**: `DeviceRef`→`ResourceRef` (alias kept); `editable_hitl_tools` emptied + config-injected.
- ✅ **Stage B**: tool→HITL seam pluggable (`network_batch_resolver.py` via `batch_resolver_fn`); neutral coreferencer default. Stage C → P2-4.

## HITL — 2026-05
- ✅ **H2 async fire-and-forget**: `request_approval_async()`, inject queue, SSE notify, SLA timeout, operator-approve → follow-up turn. Remaining → P1-7, P2-2, P2-3.
- ✅ action_type enum/builder.

## Tech-debt sweep + infra — 2026-05
- ✅ legacy LangGraph `hitl/*` removed (`HITL_BACKEND=core`); pending_hitl metric fixed.
- ✅ `StopOutcome.USER_CANCELLED` + WebUI Stop button (preserves partial answer).
- ✅ `auto_evolve_apply` switch (→ P3-9). `context_budget.strategy="priority"` wired.
- ✅ **loop.py decomposition**: 3428→~3250, `_stream_impl` 1448→~875; helpers/types/context + phase methods. Tests.
- ✅ skill feedback path fixed; Sprint-3-pre (tracing/checkpoint/A-B bench).
- ✅ **C1** `/metrics`; **C2** OTel auto-instrumentation; **D1** LLM semaphore.
- ✅ Cross-agent fault-scenario mock tools (alice/CRM).
