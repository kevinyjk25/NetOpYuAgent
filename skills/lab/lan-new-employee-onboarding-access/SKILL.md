---
name: lan-new-employee-onboarding-access
description: Provision and independently verify a new employee's LAN admission, delegated IDC application access, and real end-to-end application path in the reviewed local Containerlab topology.
metadata:
  skill_id: lan_new_employee_onboarding_access
  display_name: Lab New-Employee End-to-End Access Provisioning
  purpose: Exercise the reviewed L1 and L0 access workflow against real container namespaces and traffic.
  risk_level: high
  requires_hitl: 'true'
  profiles: lan
  lab_capability: access
  tags: lab,onboarding,access,cross-agent,nac,rbac,verification
  tool_deps: list_users,get_user_access,check_nac_policy,grant_user_access,lab_get_enforcement_path
  returns: Verified LAN admission, delegated IDC role, approval references, HTTP evidence, and a fully resolved observed path.
---

# Lab new-employee access

This is an ordered workflow. Never infer success from a write response.

1. Require explicit `user_id` and `app_id`. Use `list_users` and stop if the declared identity is absent or not active.
2. Read `get_user_access(user_id)`. If blocked, read `check_nac_policy(user_id)`, then call `grant_user_access` once with the explicit user and audit reason. The Network Runtime approval card is authoritative. Continue only after verified success.
3. Delegate exactly once to `dc-agent` with a self-contained prompt: `Invoke dc-app-access-diagnose for user_id=<user_id>, app_id=<app_id>. Check access and ACL; if denied grant the reviewed base role with reason=<reason>.`
4. If delegation returns `input-required`, resume that exact interrupt once after operator approval. Do not create a second plan.
5. After LAN and IDC grants are verified, call `lab_get_enforcement_path` locally for
   the same exact `user_id` and `app_id`. Do not delegate path inference and do not read
   endpoint IDs through device tools. Continue only when the returned traceroute has
   `ok=true`, `destination_verified=true`, `all_hops_resolved=true`, and every hop has
   `adjacency_verified=true`.
6. Report each phase, immutable plan and approval reference, HTTP verdict, exact node
   and link path, and the two actual simulation enforcement implementations. Any
   rejected, failed, unresolved, or unverified phase means end-to-end provisioning is
   incomplete.

## Parameters
- `user_id`: Exact user identifier declared by the lab manifest.
- `app_id`: Exact application identifier declared by the lab manifest.
