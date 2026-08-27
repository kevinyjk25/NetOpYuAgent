# NetOpYuAgent — DSH Network Operations Plugin

NetOpYuAgent is now a DeepSeek Harness-only project. DSH owns the agent loop,
Web UI, sessions, model calls, approvals, skills and subagent lifecycle.
This repository supplies network-domain tools, scoped memory, A2A routing,
durable HITL, offline evaluation and the Python bridge.

The historical FastAPI/WebUI, custom runtime loop, inbound A2A server,
legacy HITL framework and rollback launcher have been removed. Future
development targets the DSH plugin and adapter only.

## Architecture

- `dsh-plugin-netopyu/src/index.js` — DSH composition and tool registration.
- `dsh-plugin-netopyu/src/bridge.js` — persistent Worker and subprocess transport.
- `dsh-plugin-netopyu/src/a2a.js` — DSH remote-agent provider and durable continuation tools.
- `dsh-plugin-netopyu/src/hitl-store.js` — SQLite approval, grant, batch, trajectory and continuation state.
- `dsh_adapter/` — Python manifest, tool invocation, memory, skills, learning and evaluation.
- `profiles/` — isolated LAN, DC, WAN and default network capabilities.
- `tools/` — shared paging and pragmatic Netmiko/NAPALM tools.
- `agent_memory/` — scoped persistent memory reused by the DSH plugin.

The default backend is `mock`. Pragmatic mode is fail-closed and requires an
explicit device, MCP or OpenAPI source.

## Local installation

Requirements:

- Node.js `^22.19.0 || >=24`
- pnpm
- Python 3.11+
- Ollama with at least one configured local model

```bash
cd /Users/steven/NetOpYuAgent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama serve
ollama pull qwen3.5:27b
ollama pull qwen2.5:7b

scripts/netopyu-dsh install
scripts/netopyu-dsh doctor
scripts/netopyu-dsh start
```

Open <http://127.0.0.1:3080/>.

For mock-mode development and CI without device drivers:

```bash
pip install -r requirements-dev.txt
```

Dependency groups are intentionally separated: `requirements-core.txt` for
the DSH bridge, `requirements-pragmatic.txt` for real network drivers,
`requirements-observability.txt` for tracing, and `requirements-dev.txt` for
the test gate. The supported runtime matrix is recorded in
`dsh-plugin-netopyu/compatibility.json`.

For lower-latency network sessions:

```bash
scripts/netopyu-dsh settings-sync
scripts/netopyu-dsh model qwen2.5:7b
scripts/netopyu-dsh preset minimal
scripts/netopyu-dsh restart
```

## Operations

```bash
scripts/netopyu-dsh status
scripts/netopyu-dsh worker-status
scripts/netopyu-dsh worker-start
scripts/netopyu-dsh worker-stop
scripts/netopyu-dsh logs
scripts/netopyu-dsh models
scripts/netopyu-dsh backend
scripts/netopyu-dsh peers
scripts/netopyu-dsh parity
scripts/netopyu-dsh reliability
scripts/netopyu-dsh retirement
```

The launcher stores mutable runtime state under
`~/Library/Application Support/NetOpYuAgent/dsh-runtime` by default.
Use `NETOPYU_DSH_RUNTIME` to override it.

The launcher PID file and runtime/port identity are cross-checked. Status still
works in restricted environments where process inspection is allowed but
`kill -0` is denied.

## Safety

Read-only tools are exposed by default. Mutating tools require all three gates:

1. `NETOPYU_DSH_ENABLE_DESTRUCTIVE=1` at deployment time.
2. A fresh DSH `allowed-once` decision.
3. Successful consumption of the exact one-shot Tool Guard grant by the Python bridge.

Never set `NETOPYU_DSH_ALLOW_DESTRUCTIVE` manually. The plugin supplies the
per-request authorization only after DSH approval.

Large results are stored in SQLite and returned as bounded `[STORED:...]`
references. Remote A2A `input-required` states become durable continuations;
resume/reject requires another fresh DSH approval and survives plugin restart.

## A2A peers

```bash
NETOPYU_DSH_A2A_PEERS=http://127.0.0.1:8001,http://127.0.0.1:8002 \
  scripts/netopyu-dsh peers
```

Peers must expose an AgentCard and an A2A SSE stream endpoint. DSH sends only
the self-contained delegated prompt and bounded provenance metadata; parent
conversation history is not inherited.

## Offline reviewed learning

```bash
scripts/netopyu-dsh learning-mine
scripts/netopyu-dsh learning-mine --apply
scripts/netopyu-dsh learning-review workflow-... approve local-reviewer
scripts/netopyu-dsh learning-review workflow-... reject local-reviewer
```

Mining reads tool names only—never prompts, argument values or results.
Approval creates an uninstalled proposal; no generated skill is automatically
promoted into an active profile.

## Verification

```bash
scripts/netopyu-dsh doctor
scripts/netopyu-dsh retirement
```

`retirement` is the complete local gate: Python tests, DSH-only architecture
audit, Node syntax, 39-tool HITL/A2A smoke, profile skill projection, retrieval
quality, concurrent Worker load, malformed-input isolation, restart recovery,
destructive-policy enforcement and verification that retired harness surfaces
remain absent.

See `DSH_MIGRATION.md` for detailed runtime behavior and
`MIGRATION_AUDIT.md` for the latest evidence.

The next architecture layer is the deterministic L1 Network Runtime described
in `NETWORK_RUNTIME.md`. DSH remains L0; network parameter compilation,
preflight, plan-bound execution, independent verification and rollback belong
to L1.
