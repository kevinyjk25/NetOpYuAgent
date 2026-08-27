# NetOpYuAgent — DSH Network Operations Plugin

NetOpYuAgent is now a DeepSeek Harness-only project. DSH owns the agent loop,
Web UI, sessions, model calls, approvals, skills and subagent lifecycle.
This repository supplies the network-domain runtime above DSH: generalized L1
Skills, deterministic Network L0 Skills, network tools, scoped memory, A2A
routing, durable HITL, offline evaluation and the Python bridge.

The historical FastAPI/WebUI, custom runtime loop, inbound A2A server,
legacy HITL framework and rollback launcher have been removed. Future
development targets the DSH plugin and adapter only.

## Architecture

- `dsh-plugin-netopyu/src/index.js` — DSH composition and tool registration.
- `dsh-plugin-netopyu/src/bridge.js` — persistent Worker and subprocess transport.
- `dsh-plugin-netopyu/src/a2a.js` — DSH remote-agent provider and durable continuation tools.
- `dsh-plugin-netopyu/src/hitl-store.js` — SQLite approval, grant, batch, trajectory and continuation state.
- `dsh_adapter/` — Python manifest, tool invocation, memory, skills, learning and evaluation.
- `network_runtime/` — strict compiler, immutable plans, workflow guard,
  one-shot execution, typed verification, compensation and SQLite journal.
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

For model configuration:

```bash
scripts/netopyu-dsh settings-sync
scripts/netopyu-dsh model qwen3.5:27b
scripts/netopyu-dsh restart
```

Keep `qwen3.5:27b` as the default for tool-using network sessions. Local UI
qualification showed that `qwen2.5:7b` can load an L1 Skill but may render tool
names as prose instead of emitting tool calls; it is not approved for mutating
Network L0 Skill workflows. Smaller models require their own tool-call
conformance gate before use.

## Local Web UI L1 + L0 test

Start a LAN-profile mock session with approval-gated writes enabled. This only
changes the local simulator; it does not connect to a real device:

```bash
cd /Users/steven/NetOpYuAgent
scripts/netopyu-dsh stop
scripts/netopyu-dsh dc-peer-start
NETOPYU_PROFILE=lan \
NETOPYU_DSH_BACKEND=mock \
NETOPYU_DSH_ENABLE_DESTRUCTIVE=1 \
NETOPYU_DSH_A2A_PEERS=http://127.0.0.1:8765 \
scripts/netopyu-dsh start
```

Open <http://127.0.0.1:3080/>, select `qwen3.5:27b`, and create a new session.
Use this prompt:

```text
这是本地 mock 网络演练。请调用 lan-new-employee-onboarding-access Skill，
为新员工 erin 开通 CRM 的端到端访问。严格实际调用工具；所有写入都让我在
DSH 的 Network L0 Skill 计划审批卡中审批，不要使用通用提问代替审批。
```

For the LAN write, the approval card must show the exact arguments, plan hash,
intent hash, L0 Skill id/version/hash, verifier, rollback contract and workflow
binding. After `允许一次`, verify the terminal state and hash chain with:

```bash
scripts/netopyu-dsh runtime-list 5
scripts/netopyu-dsh runtime PLAN_ID
scripts/netopyu-dsh runtime-audit PLAN_ID
```

The local `dc-agent` is loopback-only and mock-only. It exercises the real A2A
AgentCard/SSE transport, reviewed DC workflow, immutable DC Network L0 plan,
durable DSH continuation approval, independent verification and final path
check. It refuses pragmatic mode. Confirm discovery with
`scripts/netopyu-dsh dc-peer-status` and `scripts/netopyu-dsh peers`.
The peer deliberately executes reviewed DC Skill semantics deterministically;
the parent DSH session remains the local-LLM L1 orchestrator. A production
multi-agent deployment should replace it with a separately operated DC DSH
agent. The offline `scripts/netopyu-dsh demo-l1-l0` command remains the
deterministic two-domain regression, but is not a substitute for the Web UI test.

## Operations

```bash
scripts/netopyu-dsh status
scripts/netopyu-dsh worker-status
scripts/netopyu-dsh dc-peer-start
scripts/netopyu-dsh dc-peer-status
scripts/netopyu-dsh dc-peer-stop
scripts/netopyu-dsh worker-start
scripts/netopyu-dsh worker-stop
scripts/netopyu-dsh logs
scripts/netopyu-dsh models
scripts/netopyu-dsh backend
scripts/netopyu-dsh runtime-list
scripts/netopyu-dsh runtime PLAN_ID
scripts/netopyu-dsh l0-skills
scripts/netopyu-dsh demo-l1-l0
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

Read-only tools are exposed by default. Mutating tools require every gate:

1. `NETOPYU_DSH_ENABLE_DESTRUCTIVE=1` at deployment time.
2. A fresh DSH `allowed-once` decision.
3. A strictly compiled immutable L1 plan with fresh preflight evidence.
4. Successful consumption of a Tool Guard grant bound to the exact plan hash.
5. A typed independent postcondition read before success can be returned.

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
audit, Node syntax, 40-tool HITL/A2A/runtime smoke, profile skill projection, retrieval
quality, concurrent Worker load, malformed-input isolation, restart recovery,
destructive-policy enforcement and verification that retired harness surfaces
remain absent.

See `DSH_MIGRATION.md` for detailed runtime behavior and
`MIGRATION_AUDIT.md` for the latest evidence.

The domain Network Runtime is implemented as described in `NETWORK_RUNTIME.md`.
DSH replaces the base agent-framework layer; within the network domain, L1
Skills handle generalized reasoning and Network L0 Skills own deterministic
effects. Parameter compilation, preflight, plan-bound execution, independent
verification, recovery and rollback therefore remain in this repository. The
P1 foundation also rechecks target state immediately before a write, dispatches
verification/compensation through versioned contracts and exposes
`scripts/netopyu-dsh runtime-audit PLAN_ID` for event hash-chain verification.
Every local mutating capability is additionally exposed through a versioned
Network L0 Skill contract. Inspect its fixed steps, intent type and failure
policy with `scripts/netopyu-dsh l0-skills`; raw write preparation without the
exact L0 Skill binding is rejected. Run `scripts/netopyu-dsh demo-l1-l0` for an
isolated walkthrough in which LAN and DC L1 Skills diagnose one access problem,
invoke two L0 Skills, verify the final state and audit both event chains. See
`L1_L0_SKILL_DEMO.md` for the stage-by-stage review and trust boundaries.
