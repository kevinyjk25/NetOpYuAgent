# NetOpYuAgent → DeepSeek Harness migration

This repository now contains the complete local DSH deployment migration. DSH owns
the agent loop, session log, tool execution pipeline and UI. Existing NetOpYu
profile callables remain behind a narrow Python subprocess bridge as the domain
implementation; they are plugins of DSH rather than a second agent runtime.

## Current scope

- `dsh-plugin-netopyu/` is an installable DSH bundle.
- The plugin is split by responsibility: `src/index.js` owns DSH composition,
  `src/bridge.js` owns Worker/subprocess transport, `src/a2a.js` owns the remote
  provider and continuation tools, and `src/hitl-store.js` owns SQLite/HITL and
  one-shot Tool Guard state. The package manifest includes every source module.
- `dsh_adapter/` exports profile metadata and invokes existing Python tools.
- Only `read_only` tools are registered by default.
- Large tool results preserve the legacy context-budget contract under DSH:
  payloads over 4,000 characters are written to a durable SQLite store and the
  model receives a `[STORED:tool:id]` reference. Every profile exposes
  `read_stored_result` and `process_stored_chunks`; references survive the
  short-lived Python bridge process used for each tool call.
- Setting `NETOPYU_DSH_ENABLE_DESTRUCTIVE=1` only registers mutating tools.
  Before a write, L1 strictly compiles and preflights an immutable plan; DSH
  approves the exact plan summary; Tool Guard binds its one-shot grant to the
  plan hash; L1 executes once and returns success only after typed independent
  verification. Direct bridge writes are retired.
- Approval arguments and outcomes are recorded in `data/dsh_hitl.sqlite`.
  Startup marks interrupted `pending`/`approved`/`resuming` rows as `orphaned`.
  The read-only `netopyu_hitl_list` tool lists recoverable operations, and
  `netopyu_hitl_resume` resubmits one with its original or fully replaced
  arguments after a new DSH `allowed-once` decision. Changed argument keys are
  restricted to the legacy `editable_hitl_tools` allow-list. A pre-restart
  approval is never reused.
- `netopyu_hitl_batch` maps legacy batch approval to one DSH approval and
  persists every item as queued/running/completed/failed/skipped. It supports
  `best_effort` and fail-fast `all_or_nothing` policies.
- `netopyu_hitl_async_submit` maps non-blocking HITL to a durable `deferred`
  request: it immediately returns the caller's optimistic default and performs
  no network action. A later `netopyu_hitl_resume` call needs fresh approval;
  unclaimed requests expire at their SLA.
- `NetOpYuToolGuard` is provided on the Cordis context as
  `netopyuToolGuard`. Approval grants are stored in SQLite using only a
  SHA-256 execution-token digest. Its monotonic lifecycle is
  `issued -> consumed | revoked | orphaned`; conditional updates guarantee a
  grant can be consumed at most once, including concurrent/replayed calls.
- `NetOpYuMemoryService` is provided as `netopyuMemory`; its
  `netopyu_memory_recall` tool derives the session id from the live DSH
  execution and binds the operator from deployment configuration. Recall is
  strict to that operator+session pair and is never injected automatically.
- `NetOpYuCapabilityService` is provided as `netopyuCapabilities`; its
  `netopyu_capability_search` tool reuses the CJK-aware BM25 retrieval corpus
  for tools and skills. Tool results are intersected with the tools actually
  exposed by the current DSH profile, so disabled mutating tools cannot leak
  back through discovery.
- `NetOpYuA2AProvider` is registered on DSH as the remote subagent provider
  `netopyu-a2a`. `netopyu_delegate` routes an explicit agent id or advertised
  capability to an A2A `message/stream` endpoint, while
  `netopyu_peer_list` exposes discovery diagnostics. Remote runs never inherit
  parent conversation history; only the self-contained prompt and bounded
  provenance metadata cross the process boundary. Timeouts, unreachable peers,
  and self/chain loops fail closed instead of being reported as successful
  execution. Peer `input-required` states also fail the current call closed,
  while persisting a restart-safe continuation that can only be resumed or
  rejected through a fresh DSH approval.
- DSH session/tool lifecycle events are persisted in the HITL SQLite database
  as privacy-minimized trajectories (event type, sequence, tool name, argument
  key names, and outcome only; no prompt text or argument values). The read-only
  `netopyu_trajectory_recent` tool makes this data available to evaluation and
  workflow mining without reviving the legacy WebUI journal.

Set `NETOPYU_DSH_HITL_STORE` to override the SQLite location for tests or a
deployment-managed persistent volume.
Set `NETOPYU_DSH_TOOL_RESULT_STORE` to override the large-result store (default:
`<DSH runtime>/data/tool_results.sqlite` when using `scripts/netopyu-dsh`).
Set `NETOPYU_DSH_MEMORY_DIR` to the legacy agent memory directory to recall
existing data (default: `data/agents/<profile>-agent/memory`). Set
`NETOPYU_DSH_OPERATOR_ID` to the authenticated deployment identity (default:
`dev-user` for compatibility with the local legacy configuration). A missing
memory database returns an empty diagnostic without creating files.
- Canonical `SKILL.md` files remain under `skills/` and `profiles/<profile>/skills/`.
  At plugin startup, `dsh_adapter.skills` projects exactly the active
  `NETOPYU_PROFILE` catalog into DSH's runtime skill registry, including the
  common `read-stored-result` skill and profile-relative scripts/references.
  There is no generated `.dsh/skills` copy to drift or leak LAN skills into a
  DC/WAN agent.

## Recommended local run

DeepSeek Harness currently requires Node.js `^22.19.0 || >=24` and pnpm.
The repository-local launcher pins DSH `0.1.1-rc.2`, creates an isolated
`DSH_HOME`, installs the NetOpYu bundle into the built-in Web profile, and
configures the existing Ollama `qwen3.5:27b` model.

```bash
cd /Users/steven/NetOpYuAgent
scripts/netopyu-dsh install
scripts/netopyu-dsh doctor
scripts/netopyu-dsh start
```

Open <http://127.0.0.1:3080>. The launcher starts the DSH process from this
repository and exports `NETOPYU_ROOT` so the plugin always resolves the
canonical NetOpYu code here. The workspace selected for an individual DSH
session is independent (for example `/Users/steven/Documents/DSH`) and may be
changed in the UI without changing the plugin root. The plugin registers the
selected profile's tools and skills dynamically. The pinned runtime
manifest stays in `.netopyu-dsh/`; installed packages, settings, logs, PID state
and the HITL database stay outside the watched workspace under
`~/Library/Application Support/NetOpYuAgent/dsh-runtime`. Set
`NETOPYU_DSH_RUNTIME` to choose another state directory.

Useful lifecycle commands:

```bash
scripts/netopyu-dsh status
scripts/netopyu-dsh worker-status
scripts/netopyu-dsh worker-start
scripts/netopyu-dsh worker-stop
scripts/netopyu-dsh runtime-list
scripts/netopyu-dsh runtime PLAN_ID
scripts/netopyu-dsh logs
scripts/netopyu-dsh stop
scripts/netopyu-dsh dump-config
scripts/netopyu-dsh peers
scripts/netopyu-dsh parity
scripts/netopyu-dsh learning-mine
scripts/netopyu-dsh reliability
scripts/netopyu-dsh retirement
scripts/netopyu-dsh models
```

### Offline reviewed learning

`scripts/netopyu-dsh learning-mine` reads the privacy-minimized trajectory
table in read-only mode and reports repeated tool sequences. It never reads
prompts, argument values, or results and does not create a learning database.
Add `--apply` to store idempotent `pending` candidates. This still changes no
canonical skill.

Review is explicit and one-shot:

```bash
scripts/netopyu-dsh learning-review workflow-... approve local-reviewer
scripts/netopyu-dsh learning-review workflow-... reject local-reviewer
```

Approval writes an uninstalled `SKILL.md` proposal plus review metadata under
the DSH runtime. It never registers or copies the proposal into canonical
profile skills; promotion remains an ordinary reviewed source change.

### DSH-only retirement gate

`scripts/netopyu-dsh reliability` creates isolated temporary Worker state and
tests concurrent tool calls, a one-second p95 latency ceiling, malformed input
isolation, clean Worker restart, explicit destructive request gating, the
absence of all retired harness surfaces, and presence of the DSH launcher. It
forces the mock backend and reports `real_network_actions: 0`.
`scripts/netopyu-dsh retirement` runs this rehearsal as part of the complete
offline E2E gate.

### Local model and compact-tool profiles

`settings-sync` adds both the default `qwen3.5:27b` reasoning model and the
recommended non-thinking `qwen2.5:7b` tool model without overwriting unrelated
DSH settings or silently changing the current default. Switching is explicit:

```bash
ollama pull qwen2.5:7b
scripts/netopyu-dsh settings-sync
scripts/netopyu-dsh model qwen2.5:7b
scripts/netopyu-dsh preset minimal
scripts/netopyu-dsh restart
```

To reduce the tool-schema prompt for a bounded deployment, project only named
bridge tools. Names are validated at startup and auxiliary safety/lookup tools
remain available:

```bash
NETOPYU_DSH_TOOL_ALLOWLIST=list_devices,device_info,read_stored_result,process_stored_chunks \
  scripts/netopyu-dsh start
```

Omit the variable for the complete profile. The allowlist is a deployment
optimization, not an authorization boundary; mutating tools still require
`NETOPYU_DSH_ENABLE_DESTRUCTIVE=1` plus the normal durable HITL gates.

### Persistent Python bridge

`scripts/netopyu-dsh start` and `foreground` ensure the local bridge Worker is
running first. It listens on an owner-only Unix Socket (mode `0600`) under the
DSH runtime and handles concurrent line-delimited JSON requests. Every response
is correlated by request id; cancellation closes the client connection.
Mutating invocations carry the plan id, plan hash, one-shot nonce and durable
DSH approval identity rather than changing process-global environment state.
`NETOPYU_DSH_WORKER_CONCURRENCY` bounds in-flight requests from 1–64 and
defaults to 8.

Each DSH tool call propagates its call id as a correlation id through the Node
plugin and Worker. The Worker writes one JSON log row to
`<runtime>/logs/bridge-worker.log` with request/correlation ids, command,
profile, tool, outcome, error type and duration. Prompt text and tool argument
values are deliberately excluded.

Optional OpenTelemetry spans reuse the existing Python tracing adapter and are
disabled by default. Set `NETOPYU_DSH_OTEL_ENABLED=true`; optionally set
`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SAMPLE_RATIO`, `OTEL_SERVICE_NAME`, and
`OTEL_SERVICE_VERSION`. Each `netopyu.dsh.bridge` span contains only command,
profile, tool and correlation identifiers. Missing OTel packages/exporters
degrade to the existing no-op/console behavior and never prevent Worker startup.

Override the location with `NETOPYU_DSH_WORKER_SOCKET`. If no Worker owns that
Socket, the Node plugin falls back to the original short-lived subprocess
transport. A local 10-call mock benchmark measured approximately 15 ms/call
through the Worker versus 457 ms/call through subprocess startup (30.4x).

The `minimal` DSH preset is the recommended network-operations route: it keeps
globally registered NetOpYu tools but replaces the full coding-agent prompt and
built-in catalog with DSH's short fixed persona and two local coding tools.
Use `standard` when shell/filesystem/search/skills/planning/subagent features
are needed. Preset changes apply only to new sessions after restart.

### Remote network agents (A2A subagents)

Configure peer base URLs either in `config.yaml` under `agent.peers`, through
the legacy `AGENT_PEERS` variable, or with the DSH-specific override:

```bash
NETOPYU_DSH_A2A_PEERS=http://127.0.0.1:8001,http://127.0.0.1:8002 \
  scripts/netopyu-dsh doctor
NETOPYU_DSH_A2A_PEERS=http://127.0.0.1:8001,http://127.0.0.1:8002 \
  scripts/netopyu-dsh start
```

The peers must expose an A2A AgentCard and the `POST <card-url>/stream` SSE
endpoint. Use `scripts/netopyu-dsh peers` to verify discovery before asking DSH
to call `netopyu_delegate`. `target` selects an exact `agent_id`/card name;
`capability` selects from advertised skill ids, names, tags, and descriptions.
If a peer returns `input-required`, the DSH call stops with a remote-HITL
diagnostic and stores a durable continuation in the plugin HITL database. Call
the read-only `netopyu_a2a_hitl_list` tool to inspect waiting/failed
continuations, then call `netopyu_a2a_hitl_resume` with the continuation id and
an `approve` or `reject` decision. Resume is a new DSH write operation and
therefore always requires a fresh `allowed-once` approval, including after a
plugin restart. The original prompt and routing metadata are replayed from the
store; the plugin never treats the earlier peer interrupt as authorization.
For local simulation, the peer recognizes the bounded `resume_interrupt_id`
and `operator_decision` A2A metadata and returns the continued task result.

`NETOPYU_DSH_A2A_TIMEOUT` controls the remote stream timeout (default 300s),
and `NETOPYU_DSH_A2A_MAX_HOPS` bounds the propagated delegation chain (default
3). With no configured peers DSH remains fully usable in solo mode and remote
delegation explicitly reports unavailable.

Mutating tools remain disabled by default. To expose them behind DSH approval
and the L1 Network Runtime:

```bash
NETOPYU_DSH_ENABLE_DESTRUCTIVE=1 scripts/netopyu-dsh start
```

Do not set `NETOPYU_DSH_ALLOW_DESTRUCTIVE=1` yourself. The plugin supplies the
execution permission only for `runtime-execute`, after DSH has issued an
`allowed-once` grant. The Tool Guard binds that grant to the exact DSH
execution token, tool name and immutable plan hash;
unused grants are revoked after execution and all still-issued grants become
`orphaned` at plugin restart.

The Worker marks any crash-interrupted write indeterminate and immediately
reconciles it with verifier reads only. It never replays the write. A proven
postcondition becomes `verified_success`; otherwise the durable state becomes
`manual_intervention_required`. Inspect recent plans with `runtime-list` and a
complete state/evidence journal with `runtime PLAN_ID`.
Use `runtime-audit PLAN_ID` to verify the per-plan event hash chain; a mismatch
is a hard audit failure and is never repaired automatically.
Use `l0-skills` to inspect the deterministic Network L0 Skill catalog. DSH
injects the manifest-provided L0 Skill id into single, recovery and batch plan
preparation; callers cannot substitute an arbitrary tool or skill id.

### Recover an interrupted operation

Start DSH with mutating tools enabled, then ask the agent to call
`netopyu_hitl_list`. For the selected request, call `netopyu_hitl_resume` with
its `request_id`. Its optional `arguments` field is a **full replacement** for
the recorded arguments, not a partial merge, and every changed/removed/added
key must be in that tool's configured editable allow-list. DSH will display a
new approval request before any network operation runs.

This is explicit new-turn resubmission rather than transparent continuation of
the pre-crash tool call: DSH's in-turn approval promise does not survive process
restart. The SQLite audit links the recovery approval to the original request,
uses an atomic claim to prevent duplicate execution, and returns a failed
recovery to the recoverable list.

### Batch and asynchronous approval

Call `netopyu_hitl_batch` with 1-50 `operations` and `policy` set to
`best_effort` or `all_or_nothing`. Every operation must reference an enabled
approval-gated tool. `all_or_nothing` means validate-first and stop-on-first
failure; it cannot transactionally undo network actions that already completed.

Call `netopyu_hitl_async_submit` with `tool_name`, `arguments`,
`default_value`, and an optional `sla_seconds` (60-86400). The returned
`request_id` appears in `netopyu_hitl_list` until it is resumed or expires.
Submitting is safe and non-blocking because it does not invoke the backend.

### Real network backend

The default backend remains `mock`. Select the production-oriented backend only
after configuring at least one real source in `config.yaml`:

- `pragmatic.device_inventory` for Netmiko/NAPALM device access;
- `pragmatic.mcp_servers` or `MCP_CONFIG_JSON` for MCP servers;
- `tools.openapi.spec_url` plus `tools.openapi.base_url` for a NetOps REST API.

```bash
NETOPYU_DSH_BACKEND=pragmatic scripts/netopyu-dsh doctor
NETOPYU_DSH_BACKEND=pragmatic scripts/netopyu-dsh start
```

Setting top-level `mode: pragmatic` in the selected `NETOPYU_CONFIG_PATH` is
equivalent; the environment variable is the convenient one-command override.

Pragmatic mode is fail-closed: it never loads the built-in mock MCP/OpenAPI
fallback. `doctor` fails when no real source is configured, and the backend
inventory can be inspected independently with:

```bash
NETOPYU_DSH_BACKEND=pragmatic scripts/netopyu-dsh backend
```

## Development run from a DSH checkout

```bash
cd /path/to/deepseek-harness
pnpm install
pnpm run build
pnpm dsh web --no-open \
  --patch /Users/steven/NetOpYuAgent/dsh-plugin-netopyu/cordis.patch.local.yml
```

Run the command from `/Users/steven/NetOpYuAgent` so the bridge can resolve the
canonical profile tools, skills, scripts, references, configuration, and data.

For a normal installed DSH, install the bundle into a profile:

```bash
dsh plugin --profile netopyu add /Users/steven/NetOpYuAgent/dsh-plugin-netopyu
cd /Users/steven/NetOpYuAgent
dsh --profile netopyu
```

Select a profile or Python interpreter with environment variables:

```bash
NETOPYU_PROFILE=dc NETOPYU_PYTHON=/path/to/python dsh --profile netopyu
```

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_dsh_adapter.py tests/test_dsh_a2a_provider.py
for source in dsh-plugin-netopyu/src/*.js; do node --check "$source"; done
node dsh-plugin-netopyu/test/hitl-smoke.mjs
bash -n scripts/netopyu-dsh
scripts/netopyu-dsh doctor
scripts/netopyu-dsh retirement
```

## Migration status

The DSH migration and hard retirement are complete. The historical FastAPI
entry point, WebUI, custom runtime loop, inbound A2A server, legacy HITL/task
orchestration, rollback launcher, obsolete learning stack and their dedicated
tests have been removed. `scripts/netopyu-dsh` is the only supported runtime
entry point. Existing memory, tool-result, approval and golden-set data remain
compatible and are not deleted by source retirement.
