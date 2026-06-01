---
name: lan-user-access-diagnose
description: Diagnose why a user cannot access an application. Determine whether the
  cause is LAN-side (identity / network admission) or application-layer (which lives
  on the DATA CENTER and must be delegated). Do NOT assume a network-path problem
  first.
metadata:
  skill_id: lan_user_access_diagnose
  display_name: LAN User Access Diagnose
  purpose: Diagnose why a user cannot access an application. Determine whether the
    cause is LAN-side (identity / network admission) or application-layer (which lives
    on the DATA CENTER and must be delegated). Do NOT assume a network-path problem
    first.
  risk_level: low
  requires_hitl: 'false'
  tags: user,access,permission,identity,nac,troubleshoot
  tool_deps: list_users,get_user_access,check_nac_policy,grant_user_access
  returns: LAN admission verdict + whether to delegate app-access check to DC
---

# LAN User Access Diagnose

Use this when a user reports they cannot access / are denied / cannot reach an application. Access failures are usually identity or permission problems, not network-reachability problems — check those FIRST and treat fabric/path diagnostics as a last resort. Procedure, in order: (1) get_user_access(user_id) — is the user admitted on the LAN (RADIUS / 802.1X / NAC / VLAN)? (2) If blocked, check_nac_policy(user_id) explains the NAC decision; restoring admission is grant_user_access (destructive, needs approval). (3) If the user IS fully admitted on the LAN, the LAN is not the cause: the problem is application-layer access control, which is owned by the DATA CENTER. DELEGATE the application-access check to the dc agent (it owns application RBAC/ACL via dc_app_access_diagnose). Describe the task plainly as 'check whether user <id> has access permission to application <app>' — do NOT pre-frame it as a VNI / overlay / BGP-EVPN routing problem. (4) Only if BOTH LAN admission and DC application access are confirmed OK should network-path reachability be investigated.

## Parameters
- `user_id`: User reporting the access failure (e.g. alice)
- `app`: Application the user cannot reach (e.g. CRM)

## Examples
- args: {'user_id': 'alice', 'app': 'CRM'} — Why can't alice reach CRM
