---
name: conditional-ticket-triage
description: Diagnose one service ticket with explicit conditional branches and no writes.
allowed-tools: ticket_get service_health_get
metadata:
  risk_level: low
---
# Conditional ticket triage

## Parameters

- `ticket_id`: Explicit ticket identifier.

## Steps

1. Read the ticket and affected service.
2. If the service is healthy and required evidence is missing, ask for that evidence.
3. If the service is degraded, read current health signals and summarize the matching symptoms.
4. If the service cannot be resolved, stop and state that the branch is unknown.
5. Never update, assign, or close the ticket.
