---
name: url1-network-access
description: Grant one active user time-bounded access to an approved VLAN through URL1; use only after confirming current access and always verify or restore the previous state.
allowed-tools: url1_grant_network_access
metadata:
  skill_id: url1_network_access
  display_name: URL1 Network Access Grant
  purpose: Safely grant one user time-bounded access to one VLAN through the reviewed URL1 API.
  risk_level: medium
  requires_hitl: 'true'
  profiles: lan
  tags: network,access,rest,l1-to-l0
  tool_deps: url1_grant_network_access
  returns: Verified access state or verified restoration state
---

# URL1 Network Access Grant

Use URL1 to grant one existing user access to an explicitly selected VLAN for a bounded duration.
Never infer the user, VLAN, duration, or reason. Read the current access state first and preserve its
revision and full snapshot. Present the exact change for approval. After the write, read through the
independent access-state endpoint and require `allowed=true`, the requested VLAN, and a newer revision.
If verification fails, restore the exact pre-change snapshot and independently verify equality.

## Exact Semantic Intent

This marked block is the exact semantic anchor. Runtime must preserve it through L0.5 and L0 and
must not infer a replacement from prose.

<!-- netopyu:semantic-intents/v1 -->
```yaml
- effectCapability: rest.url1.network-access.grant
  kind: grant_network_access
  targetFields:
    - user_id
  desiredState:
    allowed: true
    vlan_id: "${arguments.vlan_id}"
```

## Parameters

- `user_id`: Existing active user identifier.
- `vlan_id`: Explicit VLAN identifier from 1 through 4094.
- `duration_minutes`: Requested duration from 30 through 10080 minutes.
- `reason`: Human-readable business reason of 5 through 512 characters.

## Constraints

- One user and one VLAN per execution.
- Human approval is mandatory.
- An API response alone is not success; independent readback is required.
- Unknown write outcome requires read-only reconciliation, never blind retry.

## Steps

1. Validate all parameters and resolve the user through IAM.
2. Read and snapshot current access state and revision.
3. Present the immutable plan and wait for approval.
4. Revalidate the revision immediately before sending URL1 once.
5. Independently verify the requested state.
6. On failed verification, restore and verify the exact snapshot.
