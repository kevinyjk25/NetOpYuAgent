---
name: lab-deterministic-path-query
description: Answer topology, endpoint, traffic-path, and enforcement-point questions only from the typed lab graph and observed traceroute evidence.
allowed-tools: lab_get_topology_graph, lab_get_endpoint, lab_trace_path, lab_get_enforcement_path
metadata:
  skill_id: lab_deterministic_path_query
  display_name: Lab Deterministic Topology and Path Query
  purpose: Eliminate model-inferred lab wiring and prove every reported hop against the reviewed manifest.
  risk_level: low
  requires_hitl: 'false'
  profiles: lan,dc
  lab_capability: topology
  tags: lab,topology,path,endpoint,enforcement,verification,fail-closed
  tool_deps: lab_get_topology_graph,lab_get_endpoint,lab_trace_path,lab_get_enforcement_path
  returns: Typed graph facts or an observed, fully resolved path with explicit simulation boundaries.
---

# Deterministic topology and path query

Use this Skill for every question about lab wiring, endpoint attachment, traffic path,
devices traversed, security boundaries, or policy enforcement location.

## Required flow

1. Never reconstruct topology from `get_device_config`, interface descriptions, routing
   tables, or model knowledge. Call `lab_get_topology_graph` for topology questions.
2. An endpoint is not a device. Call `lab_get_endpoint(endpoint_id)` instead of a device
   tool whenever the target is a client or server.
3. For a generic endpoint-to-endpoint path, require exact `source_endpoint` and
   `destination_endpoint`, then call `lab_trace_path` exactly once.
4. For a user-to-application path, require exact `user_id` and `app_id`, then call
   `lab_get_enforcement_path` exactly once. Its traffic path and enforcement-point
   implementation are authoritative.
5. Report a path only when `ok=true`, `destination_verified=true`,
   `all_hops_resolved=true`, and every hop has `adjacency_verified=true`. Otherwise
   state that the path is unverified and include the exact unresolved evidence. Never
   fill an unknown hop with a likely device.

## Truth boundary

- `endpoint-interface-state` simulates network admission; it is not real RADIUS,
  802.1X, NAC, VLAN switching, or wireless RF behavior.
- `server-source-blackhole-route` simulates application source policy; it is not a
  leaf ACL, stateful firewall rule, or application IAM/RBAC assignment.
- `routed-wan-edge-no-stateful-firewall` means the security-edge nodes provide routed
  Internet transit only. Do not put them in an internal campus-to-IDC path unless the
  observed traceroute actually traverses them.
- All router-like lab devices forward through Linux/FRR L3 namespaces. Do not describe
  them as hardware switches, EVPN/VXLAN fabric, or physical firewall appliances.

## Parameters

- `source_endpoint`: Exact source endpoint ID from the reviewed manifest.
- `destination_endpoint`: Exact destination endpoint ID from the reviewed manifest.
- `user_id`: Exact lab user ID for enforcement-path queries.
- `app_id`: Exact lab application ID for enforcement-path queries.
