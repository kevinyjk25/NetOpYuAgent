---
name: scripted-network-change
description: Apply one approved network interface description and independently verify or restore it.
allowed-tools: network_get_interface network_set_description network_restore_interface
metadata:
  risk_level: high
  requires_hitl: 'true'
  effect-runtime-script-roles: scripts/preflight.py=preflight,scripts/apply.py=provider_adapter,scripts/verify.py=verifier,scripts/rollback.py=compensator
---
# Scripted network change

Read [the provider contract](references/provider.md). The reviewed implementation contains
`scripts/preflight.py`, `scripts/apply.py`, `scripts/verify.py`, and `scripts/rollback.py`.
Inspection must never run these scripts; the effect and rollback scripts require Capability bindings.

## Steps

1. Validate device, interface, expected revision, and description.
2. Capture the exact current interface snapshot using preflight.
3. Obtain approval for the immutable plan.
4. Apply once, then verify through an independent read path.
5. On mismatch, restore the snapshot and verify exact equality.
