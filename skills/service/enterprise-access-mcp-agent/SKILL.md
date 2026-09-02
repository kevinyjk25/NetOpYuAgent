---
name: enterprise-access-mcp-agent
description: Real-agent demonstration of reading identity, application, approval-change, and permission systems over independent MCP processes, then safely reconciling one entitlement through L0 Runtime.
metadata:
  skill_id: enterprise_access_mcp_agent
  display_name: Enterprise access MCP Agent
  purpose: Demonstrate actual Agent-to-external-system integration with separate identity, application, change approval, and access-policy MCP services sharing transactional business data.
  risk_level: high
  requires_hitl: 'true'
  profiles: lan,dc
  tags: agentized,mcp,identity,approval,permission,external-system,l1,l0
  tool_deps: identity_get_user,application_get,access_policy_evaluate,access_policy_get_entitlement,change_validate_window,access_policy_grant_entitlement
  returns: Source-attributed external-system facts plus an independently verified entitlement outcome or a fail-closed reason.
---

# Enterprise access MCP Agent

Use this for an exact request to inspect or grant one user's business access to
one application. In the local environment these are real, independent MCP
stdio server processes backed by one transactional SQLite business database;
their data is simulated, but their integration path is not an in-process tool
stub.

1. Require exact `user_id` and `app_id`. For a write also require exact
   `role`, `change_id`, and operator-provided `reason`; do not invent any value.
2. Call `identity_get_user` from the identity MCP system and `application_get`
   from the application MCP system. Stop when either record is unknown, the
   identity is inactive, or the role is not valid for the application.
3. Call `access_policy_evaluate` and `access_policy_get_entitlement` from the
   access-policy MCP system. Preserve the exact returned `revision` for
   optimistic concurrency. If the requested role already exists, report the
   observed state and stop without a write.
4. Call `change_validate_window` from the change/approval MCP system. The
   business change record is an external prerequisite, not a substitute for
   the DSH human approval. Stop unless it is approved and its window is open.
5. Call `access_policy_grant_entitlement` once using the exact identifiers,
   role, change id, preserved revision, and reason. The adapter must first
   compile a `service.access.entitlement.grant` L0 plan and present a DSH
   approval card. Never call an internal restore capability.
6. Success requires the Runtime's independent post-write read and terminal
   `verified_success` evidence. On verification failure, Runtime compensates
   to the exact pre-state and verifies restoration. Never retry blindly.
7. In the final report identify every source (`mcp:identity-service`,
   `mcp:application-service`, `mcp:change-service`, and
   `mcp:access-policy-service`) and distinguish external business approval
   from the one-shot DSH execution approval.

## Parameters

- `user_id`: Exact enterprise user, such as `erin`.
- `app_id`: Exact application, such as `crm`.
- `role`: Exact catalog role, such as `sales-rep`.
- `change_id`: Approved open change record, such as `CHG-1001`.
- `reason`: Operator-provided audit reason.

