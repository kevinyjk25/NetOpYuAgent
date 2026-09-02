---
name: agentized-lan-access-remediation
description: Real-agent demonstration of diagnosing and safely restoring one user's LAN admission through L1 guidance and a deterministic L0 Runtime write contract.
metadata:
  skill_id: agentized_lan_access_remediation
  display_name: Agentized LAN access remediation
  purpose: Demonstrate user prompt to LLM reasoning to L1 Skill to L0 plan, approval, effect, independent verification, and audit evidence.
  risk_level: high
  requires_hitl: 'true'
  profiles: lan
  tags: agentized,demo,lan,access,l1,l0,runtime
  tool_deps: get_user_access,check_nac_policy,grant_user_access
  returns: A phase-by-phase report that separates model decisions from deterministic Runtime evidence.
---

# Agentized LAN access remediation

Use this Skill only when the user asks to diagnose and, if necessary, restore
one exact user's LAN admission. This is an Agent use case: interpret the user's
request, but never turn ambiguity into write parameters.

1. Require an exact `user_id`. Require a user-supplied audit `reason` before a
   write; do not invent a ticket, identity, or reason.
2. Call `get_user_access` for that exact user. If already admitted, report the
   observed state and stop without a write.
3. If blocked, call `check_nac_policy`. Stop when the result says the identity
   is unknown/inactive or the cause is not safely remediable by LAN admission.
4. If restoration is appropriate, call `grant_user_access` exactly once with
   only `user_id` and `reason`. The LLM does not execute this write directly:
   the adapter compiles a `network.lan.user-access.grant` L0 plan, validates
   parameters and policy, then shows the operator a DSH approval card.
5. After approval, trust success only when the Runtime returns a terminal
   `verified_success` envelope based on independent verification. Never treat
   the provider's write message as success, and never retry an unknown result.
6. Summarize four boundaries explicitly: what the LLM inferred, which L1 Skill
   was used, which L0 contract executed, and which evidence proved the outcome.

## Parameters

- `user_id`: Exact user identifier.
- `reason`: Exact operator-provided audit reason.

## Example

“检查 `erin` 的网络准入；如果被阻止，请恢复。变更原因：`CHG-1001 新员工入职`。”

