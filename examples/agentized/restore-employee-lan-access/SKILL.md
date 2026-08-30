---
name: restore-employee-lan-access
description: Restore one active employee's LAN admission after checking the current state and policy.
allowed-tools: grant_user_access
metadata:
  skill_id: restore_employee_lan_access
  display_name: Restore employee LAN access
  purpose: Restore an active employee's LAN admission safely and verify the result.
  risk_level: high
  requires_hitl: 'true'
  profiles: lan
  tags: lan,access,remediation
  tool_deps: grant_user_access
  returns: Independently verified LAN admission or an explicit recovery state.
---

# Restore employee LAN access

Check the exact user first. Never infer a user id. Preserve the current access
state, require approval for the exact change, grant access once, and verify the
new state independently. If verification fails, restore the previous state and
verify restoration. Never retry an uncertain write blindly.

## Parameters

- `user_id`: Exact enterprise user identifier.
- `reason`: Required audit reason supplied by the operator.

## Constraints

- Human approval is mandatory.
- Stop when the identity is inactive or unknown.
- A write response is not proof of success.
