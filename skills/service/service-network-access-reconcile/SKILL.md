---
name: service-network-access-reconcile
description: Reconcile enterprise entitlement desired state with actual Containerlab enforcement and traffic, then execute separately approved Service and Network L0 changes when required.
metadata:
  skill_id: service_network_access_reconcile
  display_name: Service and Network Access Reconciliation
  purpose: Keep enterprise business authorization and network enforcement independent, observable, safely changeable, and exactly compensatable.
  risk_level: high
  requires_hitl: 'true'
  profiles: lan,dc
  tags: service,network,mcp,containerlab,reconciliation,access
  tool_deps: identity_get_user,application_get,access_policy_evaluate,access_policy_get_entitlement,change_validate_window,cmdb_get_endpoint_binding,network_get_app_enforcement,lab_app_probe,reconcile_service_network_access,access_policy_grant_entitlement,access_policy_revoke_entitlement,network_apply_app_enforcement,network_revoke_app_enforcement
  returns: Independent desired-state, enforcement, traffic, plan, approval, verification, compensation, and drift evidence.
---

# Service and network access reconciliation

Require exact `user_id`, `app_id`, `change_id`, and `reason`. Never treat an
application entitlement as proof of network reachability, and never infer a
business entitlement from a successful packet or HTTP request.

1. Read `identity_get_user`, `application_get`, `access_policy_evaluate`,
   `access_policy_get_entitlement`, both CMDB endpoint bindings, and
   `network_get_app_enforcement` for the same exact identifiers.
2. Call `reconcile_service_network_access`. Stop on an unresolved CMDB binding,
   unknown identity/application, closed change window, or ambiguous target.
3. If Service desired state is wrong, prepare exactly one
   `access_policy_grant_entitlement` or `access_policy_revoke_entitlement` L0
   plan using the freshly read revision. Wait for explicit approval and a
   verified Service MCP postcondition. Never call the internal restore tool.
4. Re-read reconciliation. If network enforcement is wrong, prepare exactly one
   `network_apply_app_enforcement` or `network_revoke_app_enforcement` L0 plan.
   Wait for explicit approval, fresh provider verification, and traffic proof.
5. Reconcile again. Success requires matching Service desired state, matching
   Containerlab enforcement, resolved CMDB bindings, and traffic consistent
   with the policy. A write response by itself is never success.
6. If verification fails, the Effect Runtime uses the registered domain
   compensator and independently proves restoration. Do not retry an
   indeterminate write.

## Parameters
- `user_id`: Exact enterprise identity.
- `app_id`: Exact application identifier.
- `change_id`: Approved open change record, such as `CHG-1001` in the local lab.
- `reason`: Operator-provided audit reason.
