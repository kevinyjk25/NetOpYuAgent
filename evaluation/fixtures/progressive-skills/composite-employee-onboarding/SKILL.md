---
name: composite-employee-onboarding
description: Orchestrate reviewed identity, ticket, and network L0 contracts for one employee.
metadata:
  risk_level: medium
  requires_hitl: 'true'
---
# Composite employee onboarding

Use [the onboarding policy](references/policy.md). This L1 Skill may call only active, versioned,
digest-bound L0 contracts. It must never call a provider write tool directly.

## Steps

1. Read the approved onboarding ticket.
2. If employment status is not active, stop and ask for correction.
3. Call active L0 `iam.account.create`, then `network.access.grant`.
4. If the second contract fails, use only the compensations declared by the completed L0 transaction.
5. Return verified outcomes and evidence ids; do not reinterpret an L0 terminal state.
