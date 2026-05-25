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


def get_batch_resolver_for_profile(profile: str):
    """L0/L1 Stage B: return the multi-target batch HITL resolver for a
    profile, or None. Network profiles (lan/dc) use the device-prose
    resolver; default (and any non-network profile) returns None → the L0
    loop raises single-target HITL (the domain-free default).

    The L0 runtime calls the returned function via its injected
    batch_resolver_fn; it never imports this module itself.
    """
    if profile in ("lan", "dc"):
        from profiles.network_batch_resolver import resolve_network_batch
        return resolve_network_batch
    return None


__all__ = [
    "Profile", "load_profile", "available_profiles",
    "get_batch_resolver_for_profile",
]
