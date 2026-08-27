# NetOpYuAgent DSH-only retirement audit

Audit date: 2026-08-27

## Verdict

Migration and hard retirement are complete for the confirmed local-simulation
scope. DeepSeek Harness is the only supported agent runtime and UI. NetOpYu
remains a DSH plugin plus a narrow Python domain bridge.

The historical FastAPI entry point, browser WebUI, custom agent loop, inbound
A2A server, legacy HITL/task/scheduler stack, rollback launcher, obsolete
learning implementation and their dedicated tests have been removed. Future
work must extend `dsh-plugin-netopyu/` or `dsh_adapter/`; the DSH-only
architecture audit prevents retired imports and paths from returning.

## Final evidence

- `scripts/netopyu-dsh retirement`: PASS.
- GitHub Actions runs the same clean-checkout DSH-only retirement gate; no
  retired scripts, runtime installation, Ollama model or network endpoint is
  required.
- Retained DSH/domain suite: 92 tests plus 4 profile subtests pass.
- DSH-only architecture audit: PASS.
- Node syntax and the combined 39-tool HITL/A2A/memory/trajectory smoke: PASS.
- Runtime skill projection: default 1, LAN 12, DC 5, WAN 1.
- Retrieval gate: Recall@3 1.00, MRR 0.95, zero failures.
- Clean-runtime local reliability: 24 requests at concurrency 8, p95 23.13 ms,
  maximum 23.18 ms.
- Malformed requests are isolated and the Worker remains healthy.
- Worker stop/start recovery succeeds and removes its Unix Socket cleanly.
- Ambient `NETOPYU_DSH_ALLOW_DESTRUCTIVE=1` cannot bypass an explicit
  per-request false gate; the explicit local simulation authorization path
  remains functional.
- Retired harness surfaces are absent and the DSH launcher is present.
- Reliability reports `real_network_actions: 0` and removes temporary state.
- Port 3080 remains HTTP 200 under its existing externally managed process; it
  was not stopped or restarted during source retirement.

## Removed surface

Hard retirement removed 183 files (about 2.9 MiB and 43,806 Python lines).
The temporary retirement archive was deleted after final verification; Git
retains the previously committed versions.

Removed categories:

- `main.py`, `webui/`, and `scripts/netopyu-legacy`;
- custom `runtime.loop`, old hooks/policies/journal/cache;
- `hitl_core/`, `task/`, `scheduler/`, and the legacy memory adapter;
- inbound `a2a/` server implementation;
- old integration adapters and LLM/embedder harness clients;
- legacy registry service/router/store, batch resolver and auto-evolver stack;
- legacy-only scripts, architecture documents and tests;
- FastAPI, Uvicorn, LangGraph/LangChain, legacy HITL/storage and FastAPI OTel
  dependencies from `requirements.txt`.

Kept because DSH actively uses them:

- profile network tools and canonical `SKILL.md` content;
- persistent scoped `agent_memory/`;
- pragmatic Netmiko/NAPALM, MCP and OpenAPI clients/router;
- outbound AgentCard discovery schemas;
- durable result paging and optional Worker OpenTelemetry tracing;
- golden-set evaluation, DSH settings and offline reviewed learning.

## Safety and data

Source retirement does not delete SQLite memory, tool-result, approval,
trajectory or golden-set data. No real network or external approval
environment was used. Previously committed retired source can be recovered
from Git, but it is no longer imported, tested or documented as a supported
runtime.

## Supported commands

```bash
scripts/netopyu-dsh install
scripts/netopyu-dsh doctor
scripts/netopyu-dsh start
scripts/netopyu-dsh retirement
scripts/netopyu-dsh learning-mine
scripts/netopyu-dsh reliability
```

There are no remaining migration stages in the agreed scope.
