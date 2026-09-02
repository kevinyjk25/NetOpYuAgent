---
name: multistep-cloud-maintenance
description: Perform one approved reversible cloud maintenance change with independent verification.
allowed-tools: cloud_get_instance cloud_update_instance cloud_restore_instance
metadata:
  risk_level: high
  requires_hitl: 'true'
---
# Multi-step cloud maintenance

Use [the immutable plan template](assets/change-template.json) when presenting the change.

## Steps

1. Validate the instance id, desired size, window, and reason.
2. Read health, configuration, revision, and a complete rollback snapshot.
3. Stop if health is degraded or the window is closed.
4. Bind the snapshot and revision into the immutable plan; obtain approval.
5. Revalidate revision, apply once, and independently read back health and size.
6. On failed verification, restore the exact snapshot and verify equality.
