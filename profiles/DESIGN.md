# profiles/ — Business Profile Layer

> Decouples domain-specific tools/skills/capabilities from the common agent
> framework. Added 2026-05 (Phase 2A).

## 1. Why this exists

Before profiles, business tools (`list_devices`, `edit_device_config`, …)
lived in `tools/mock_tools.py` and the framework imported them by name. The
generic agent loop was coupled to one business domain (enterprise LAN Cisco
gear). Adding a second domain (data-center fabric) — or running a pure
assistant with no business logic — meant editing framework files.

Profiles invert that. The framework knows only how to "load the active
profile"; it knows nothing about LAN vs DC. The dependency arrow points
**framework → profiles**, never the reverse.

## 2. The three profiles

| Profile | Tools | Skills | Use |
|---------|-------|--------|-----|
| `default` | 0 | 0 | Pure assistant + common meta tools. The decoupling proof: if the framework boots on `default`, nothing in `runtime/` / `a2a/` / `hitl_core/` secretly needs a business domain. |
| `lan` | 20 | 7 | Enterprise LAN: Cisco switches / APs / internal firewalls. Migrated from the old `tools/mock_tools.py`. |
| `dc` | 7 | 3 | Data-center fabric: spine/leaf VXLAN, BGP EVPN, load balancers, k8s overlay. |

Selected by `AGENT_PROFILE` env (or `agent.profile` in config.yaml),
defaulting to `default`.

## 3. The Profile contract

`profiles/base.py` defines:

```python
@dataclass
class Profile:
    profile_id:     str
    display_name:   str
    description:    str
    domain_tags:    list[str]               # for Phase-2B capability matching
    tool_callables: dict[str, Callable]     # name → async def tool(args) -> str
    tool_metadata:  dict[str, dict]          # name → prompt-facing declaration
    skill_defs:     dict[str, dict]          # skill_id → SOP definition
    capabilities:   list[dict]               # advertised in the AgentCard
```

Each `profiles/<id>/__init__.py` must expose a module-level `PROFILE: Profile`.
`profiles.load_profile(id)` discovers it lazily and caches it. An unknown id
falls back to `default` (safe degradation, logged).

## 4. Directory layout

```
profiles/
├── __init__.py        load_profile / available_profiles
├── base.py            Profile dataclass + lazy registry
├── default/
│   └── __init__.py    PROFILE (empty business)
├── lan/
│   ├── __init__.py    PROFILE (assembles the below)
│   ├── tools.py       callables (migrated from tools/mock_tools.py)
│   ├── tool_meta.py   prompt-facing metadata
│   └── skills.py      LAN SOPs
└── dc/
    ├── __init__.py
    ├── tools.py       dc_list_fabric, dc_bgp_evpn_status, …
    ├── tool_meta.py
    └── skills.py
```

## 5. How the framework consumes a profile

- `config.py` reads `AGENT_PROFILE` → `cfg.agent.profile`.
- `ToolLoader(mode, profile)` merges the common builtin tools
  (`tools/builtin/registry.py`) with `profile.tool_callables` /
  `profile.tool_metadata`. In `mock` mode the business tools come from the
  profile; in `pragmatic` mode the real device tools come from
  `tools/pragmatic_tools.py` (not yet profile-split — see TODO.md).
- `SkillLoader(mode, profile)` merges builtin skills with
  `profile.skill_defs`.
- `main.py:build_services` enriches `cfg.agent.capabilities` /
  `display_name` from the profile when the operator left them at defaults,
  so each profile advertises a sensible identity without yaml boilerplate.

## 6. Role isolation (the point)

A `lan` agent's tool registry contains only LAN tools; a `dc` agent's
contains only DC tools. A LAN agent that emits `[TOOL:dc_bgp_evpn_status]`
gets "tool not found" (the runtime's fuzzy-match suggester). The only way to
do cross-domain work is to **delegate** to the peer agent over A2A — which is
Phase 2B. Profiles are the precondition: without isolated tool sets,
delegation would be pointless (each agent would already have everything).

`scripts/audit_profiles.py` enforces this statically:
1. Every profile's `tool_callables` and `tool_metadata` keys align.
2. Business profiles have disjoint tool/skill names.
3. `default` has zero business tools/skills.
4. The framework never hard-imports `profiles.lan` / `profiles.dc` (must go
   through `load_profile`).

## 7. Common vs business — the split

| Concern | Lives in | Why |
|---------|----------|-----|
| `read_stored_result`, `process_stored_chunks` | `tools/common_tools.py` + `tools/builtin/registry.py` | Paging mechanism for large tool outputs — every profile needs it. |
| `_ts()` timestamp helper | `tools/common_tools.py` | Shared by mock log generators across profiles. |
| `list_devices`, `edit_device_config`, … | `profiles/lan/` | LAN business. |
| `dc_bgp_evpn_status`, … | `profiles/dc/` | DC business. |
| `ToolLoader` / `SkillLoader` | `tools/` / `skills/` | Framework — profile-agnostic. |

## 8. Adding a new profile

1. `mkdir profiles/<id>` with `tools.py`, `tool_meta.py`, `skills.py`,
   `__init__.py` (exposing `PROFILE`).
2. Add `<id>` to `_KNOWN_PROFILE_IDS` in `profiles/base.py`.
3. `python scripts/audit_profiles.py` — confirms isolation + consistency.
4. Add a row to `tests/test_profiles.py` if it has domain-specific
   invariants worth pinning.

## 9. Known limitations (see TODO.md)

- **Pragmatic mode isn't profile-split yet.** Real device tools
  (`tools/pragmatic_tools.py`) load regardless of profile. A `dc` agent in
  pragmatic mode would still get the LAN Netmiko tools. Splitting pragmatic
  tooling by profile is deferred — mock mode is what the A2A validation uses.
- **Phase 2B (delegation) not yet wired.** Profiles give isolation; the
  cross-agent dispatch that exploits it is the next step.


## L1 business logic (not just tools/skills) — added Stage B (2026-05)

Profiles are no longer only "bags of tools + skills". They also supply
**business decision logic** that the L0 framework calls through injection
points, so domain rules never live in `runtime/`:

- `network_batch_resolver.py` — decides whether one destructive tool call
  should expand into a multi-target batch HITL (device-prose parsing). L0 calls
  it via `AgentRuntimeLoop(batch_resolver_fn=...)`.
- `get_batch_resolver_for_profile(profile)` (in `profiles/__init__.py`) maps
  profile → resolver (lan/dc → network resolver, default → None).

Pattern for a new domain: write a resolver function with the documented
contract, register it in `get_batch_resolver_for_profile`, done — no L0 edits.
The `default` profile returning None (→ generic single-target HITL) remains the
decoupling proof.
