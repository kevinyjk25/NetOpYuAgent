---
name: approval-access-grant
description: Grant time-bounded application access after immutable-plan approval and verify or restore it.
allowed-tools: iam_get_access iam_grant_access iam_restore_access
metadata:
  risk_level: medium
  requires_hitl: 'true'
---
# Approval-gated access grant

Never infer the subject, application, role, duration, or reason.

## Parameters

- `subject_id`: Existing active identity.
- `application_id`: Explicit application.
- `role_id`: Explicit approved role.
- `duration_minutes`: Integer from 30 through 1440.
- `reason`: Business reason.

## Steps

1. Read and preserve the current access state and revision.
2. Present an immutable change plan and wait for human approval.
3. Revalidate the revision and apply the grant once.
4. Independently read back the requested role and expiry.
5. If verification fails, restore and independently verify the exact snapshot.
