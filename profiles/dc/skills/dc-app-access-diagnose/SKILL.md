---
name: dc-app-access-diagnose
description: Diagnose why a user cannot access a data-center APPLICATION — this is
  an application-layer ACCESS/PERMISSION problem (RBAC/ACL), NOT a network path /
  VNI / BGP-EVPN problem.
metadata:
  skill_id: dc_app_access_diagnose
  display_name: DC Application Access Diagnose
  purpose: Diagnose why a user cannot access a data-center APPLICATION — this is an
    application-layer ACCESS/PERMISSION problem (RBAC/ACL), NOT a network path / VNI
    / BGP-EVPN problem.
  risk_level: low
  requires_hitl: 'false'
  tags: dc,application,access,permission,rbac,troubleshoot
  tool_deps: dc_check_user_app_access,dc_get_app_acl,dc_list_apps,dc_grant_app_access
  returns: Access verdict (allowed/denied), roles held, root cause, remediation
---

# DC Application Access Diagnose

Use this when a user reports they cannot access / are denied / cannot reach a specific application (e.g. CRM). The cause is almost always application-layer access control, so check PERMISSIONS FIRST and only fall back to network diagnostics if access is confirmed granted. Procedure, in order: (1) dc_check_user_app_access(user_id, app_id) — does the user hold any role granting access? This is the primary check and usually reveals the root cause directly. (2) If denied, dc_get_app_acl(app_id) — inspect which roles grant access and who holds them, to decide the right role to assign. (3) If the user legitimately needs access, call dc_grant_app_access with `user_id`, `app_id`, the exact reviewed `role`, and a concise `reason` derived from the explicit request. This is destructive: invoke it once and let DSH compile the Network L0 Skill plan and display the authoritative plan-bound approval. Do NOT use a generic question/ask tool merely to request approval and never treat a conversational yes/no as authorization. Only independently verified tool success permits continuation. Only if access is already granted but the app is still unreachable should you escalate to network-path diagnostics (dc_path_troubleshoot).

## Parameters
- `user_id`: User reporting the access failure (e.g. alice)
- `app_id`: Application id (e.g. crm, wiki, payroll, grafana)

## Examples
- args: {'user_id': 'alice', 'app_id': 'crm'} — Why can't alice reach CRM
