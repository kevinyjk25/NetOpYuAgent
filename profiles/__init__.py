"""
profiles/ — Business profile layer (decoupled from the common agent framework)
==============================================================================

The common framework loads exactly one profile per process, chosen by
`AGENT_PROFILE` (env) or `agent.profile` (config.yaml), defaulting to "default".

Profiles
────────
  default — no business tools/skills; pure assistant + common meta tools.
            Proves the framework runs with the business layer fully removed.
  lan     — enterprise LAN (Cisco switches / APs / firewalls).
  dc      — data-center fabric (spine/leaf, BGP EVPN, VXLAN, load balancers).

See profiles/base.py for the Profile contract.
"""
from __future__ import annotations

from profiles.base import (
    Profile,
    load_profile,
    available_profiles,
)


__all__ = [
    "Profile", "load_profile", "available_profiles",
]
