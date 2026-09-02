---
name: dc-app-access-diagnose
description: Diagnose and safely grant a declared user's access to a declared IDC application using actual container policy and HTTP reachability evidence.
metadata:
  skill_id: dc_app_access_diagnose
  display_name: Lab DC Application Access Diagnose
  purpose: Bind an application access request to reviewed manifest entities, approval, exact policy mutation, verification, and inverse rollback.
  risk_level: high
  requires_hitl: 'true'
  profiles: dc
  lab_capability: access
  tags: lab,dc,application,access,rbac,verification
  tool_deps: dc_check_user_app_access,dc_get_app_acl,dc_list_apps,dc_grant_app_access
  returns: Verified application access decision, role, plan reference, and HTTP evidence.
---

# Lab DC application access

Require exact `user_id` and `app_id`; never invent either target.

1. Call `dc_check_user_app_access` for the exact pair.
2. If denied, call `dc_get_app_acl(app_id)` and select only a role returned by that read.
3. Call `dc_grant_app_access` once with the exact user, app, reviewed role, and reason. Wait for the exact Network L0 plan approval and independently verified result.
4. If verification fails, the runtime applies the registered inverse operation and verifies restoration. Do not retry automatically.
5. If access is granted but the user reports failure, run the path workflow; application success requires the actual HTTP request from the declared user endpoint.

## Parameters
- `user_id`: Exact lab user identifier.
- `app_id`: Exact lab application identifier.
