---
name: lan-new-employee-onboarding-access
description: 'Provision and VERIFY end-to-end access for a NEW employee who needs
  to use a data-center application: confirm LAN network admission, then grant application-layer
  access (DC-owned), then confirm the path is actually reachable. This is a multi-stage
  provisioning workflow that spans BOTH the LAN agent (network admission) and the
  DC agent (application RBAC + fabric path). It is NOT a single permission check.'
metadata:
  skill_id: lan_new_employee_onboarding_access
  display_name: New-Employee End-to-End Access Provisioning
  purpose: 'Provision and VERIFY end-to-end access for a NEW employee who needs to
    use a data-center application: confirm LAN network admission, then grant application-layer
    access (DC-owned), then confirm the path is actually reachable. This is a multi-stage
    provisioning workflow that spans BOTH the LAN agent (network admission) and the
    DC agent (application RBAC + fabric path). It is NOT a single permission check.'
  risk_level: medium
  requires_hitl: 'true'
  tags: onboarding,provisioning,access,cross-agent,workflow,employee,rbac,nac
  tool_deps: list_users,get_user_access,check_nac_policy,grant_user_access
  returns: 'End-to-end provisioning report: account/admission/app-role/reachability,
    with HITL approval references and per-phase status.'
---

# New-Employee End-to-End Access Provisioning

Use this for onboarding/provisioning requests like '为新员工 <user> 开通对应用 <app> 的访问' / 'set up access to <app> for new hire <user>'. This is a STRICT, ORDERED, multi-phase workflow — do not skip phases, do not jump straight to granting, and do not collapse it into one tool call. Run the phases IN ORDER and stop early only on the documented exit conditions.

PHASE 1 — LAN identity & admission (do this FIRST, locally):
  1a. list_users — confirm the user account exists AND its status is active. If it does NOT exist, STOP and report that the account must be created in IAM first. If it is suspended or disabled, STOP and require IAM remediation. Never use a network grant to bypass an inactive identity.
  1b. get_user_access(user_id) — is the user admitted on the LAN (RADIUS / 802.1X / NAC / VLAN)?
  1c. If NOT admitted: check_nac_policy(user_id) to explain why, then call grant_user_access with BOTH `user_id` and a concise `reason` derived from the explicit onboarding request. This is a destructive change: the DSH tool hook compiles the Network L0 Skill plan and automatically presents the authoritative plan-bound approval. Do NOT call a generic question/ask tool merely to request approval, and do NOT claim approval before the grant tool displays the DSH approval. Only after the approved tool returns independently verified success do you continue to Phase 2.
  1d. If already admitted: continue to Phase 2 directly.

PHASE 2 — DC application access (DELEGATE — do NOT do locally):
  Application RBAC/ACL lives on the DATA CENTER, not the LAN. Call `netopyu_delegate` exactly once with all of these fields: `target="dc-agent"`, `description="Grant <user_id> <app> application access"`, and a self-contained `prompt="Invoke dc-app-access-diagnose for user_id=<user_id>, app_id=<app>. Check current application access and ACL; if denied, grant the reviewed base role with reason=<onboarding reason>; return the verified access result and approval reference."`. Do not omit `target`, `description`, or `prompt`, and do not pass an empty value. Describe it plainly as an application-permission task — do NOT pre-frame it as a VNI / overlay / BGP-EVPN problem. The DC side may require its own operator approval (HITL) to grant a role; wait for that result before Phase 3. Do NOT re-delegate the same task to dc if it already replied. If no `dc-agent` peer is configured or reachable, STOP with Phase 2 incomplete; never report end-to-end success.

PHASE 3 — End-to-end path verification (DELEGATE — DC-owned):
  Only AFTER both LAN admission (Phase 1) and DC application access (Phase 2) are confirmed granted, call `netopyu_delegate` exactly once with `target="dc-agent"`, `description="Verify <user_id> to <app> end-to-end path"`, and a self-contained prompt that includes the verified LAN VLAN and asks the DC agent to run `dc-path-troubleshoot` for the application VIP plus load-balancer health. Do not omit `target`, `description`, or `prompt`. This is the step that distinguishes 'permission granted but still unreachable' from a fully working setup. Skip this phase only if the user explicitly said access already works and they just need the permission. If the peer is unavailable, report Phase 3 incomplete rather than inferring reachability.

PHASE 4 — Synthesize the provisioning report:
  Produce ONE final report covering: account exists (Y/N), LAN admission (was it already OK or did you restore it, with the approval reference), DC application role granted (which role, approval reference), and end-to-end reachability verdict. If any phase failed or was rejected at HITL, report exactly which phase and the consequence — do NOT claim success. Never re-run a phase that already completed; synthesize from the results already in context.

APPROVAL PROTOCOL — applies to every local or delegated write:
  If required business parameters are missing or ambiguous, ask only for those parameters. If the intent is already explicit, invoke the reviewed write tool once and let DSH produce the single authoritative Network L0 Skill approval. A conversational yes/no answer is not execution authorization. Never retry a denied or failed write automatically.

## Parameters
- `user_id`: New employee's user id (e.g. alice)
- `app`: Application to provision access for (e.g. crm)

## Examples
- args: {'user_id': 'alice', 'app': 'crm'} — Onboard alice end-to-end for CRM access
