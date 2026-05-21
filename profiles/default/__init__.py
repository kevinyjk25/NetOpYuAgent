"""
profiles/default/__init__.py — Pure assistant profile (no business)
====================================================================

The default profile carries NO business tools, NO business skills, and
advertises only generic-assistant capabilities. An agent started with
AGENT_PROFILE=default (or no AGENT_PROFILE at all) is a plain AI assistant
that only has the common framework's meta tools (read_stored_result /
process_stored_chunks, injected at runtime).

This profile exists as the proof that the business layer is cleanly
decoupled: if the framework boots and runs with the default profile, then
nothing in runtime/ / a2a/ / hitl_core/ / etc. secretly depends on a
specific business domain.
"""
from __future__ import annotations

from profiles.base import Profile

PROFILE = Profile(
    profile_id     = "default",
    display_name   = "Assistant Agent",
    description     = (
        "General-purpose assistant with no domain-specific tooling. "
        "Has only the common framework's meta capabilities."
    ),
    domain_tags    = ["general", "assistant"],
    tool_callables = {},      # no business tools
    tool_metadata  = {},
    skill_defs     = {},      # no business skills
    capabilities   = [
        {
            "skill_id":    "general_assistance",
            "name":        "General assistance",
            "description": "Answer questions and reason over provided context. "
                           "No domain-specific device/network tooling.",
            "tags":        ["general", "assistant", "qa"],
        },
    ],
)
