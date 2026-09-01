---
name: unbound-critical-purge
description: Negative fixture containing an irreversible script without a reviewed Capability binding.
metadata:
  risk_level: critical
  effect-runtime-script-roles: scripts/delete.py=provider_adapter
---
# Unbound critical purge

The package contains `scripts/delete.py`. It is intentionally not Capability-bound and must be blocked.
