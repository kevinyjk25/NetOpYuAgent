# L1 Skill + L0 Skill linked demonstration

This executable example resolves one local, simulated incident:

> New employee `erin` exists, but cannot use the `crm` application.

Run it from the repository root:

```bash
scripts/netopyu-dsh demo-l1-l0
```

The command uses fresh temporary SQLite stores and the mock backend. It performs
two simulated writes, restores the in-process simulator state afterward and
does not touch the configured DSH runtime databases or a real network.

## End-to-end path

| Stage | Owner | Decision or invariant | Report evidence |
| --- | --- | --- | --- |
| 1. Understand the request | LAN L1 Skill | Select the reviewed new-employee workflow and bind a workflow template hash | `l1_skill_selected` |
| 2. Gather prerequisites | LAN L1 Skill | Confirm the user exists, is not admitted and is blocked by NAC | `l1_prerequisites_observed` |
| 3. Compile the LAN change | LAN L0 Skill | Validate the typed intent, exact target and arguments; compile fixed steps and an immutable plan | first `l0_plan_ready` |
| 4. Authorize and execute | Network Runtime | Bind the one-time authorization to plan hash, nonce and L0 contract; recheck preflight and write once | LAN entry in `writes` |
| 5. Verify LAN state | LAN L0 Skill | Read the target independently and require the typed `admitted=true` predicate | LAN `evidence` and `verified_success` |
| 6. Hand off the remaining fault | DC L1 Skill | Start the reviewed application-access workflow with an independent template hash | `l1_skill_delegated` |
| 7. Gather DC prerequisites | DC L1 Skill | Confirm CRM still denies Erin and load the current ACL | second `l1_prerequisites_observed` |
| 8. Compile and execute RBAC | DC L0 Skill | Validate user, application, role and reason; repeat the immutable approval/write/verify sequence | second `l0_plan_ready` and DC `writes` entry |
| 9. Verify the business outcome | Independent reads | Require both LAN admission and CRM authorization, instead of treating tool return text as success | `independent_end_to_end_verification` |
| 10. Audit | Network Runtime | Recompute both append-only event hash chains and reject missing L0 bindings | `guarantees_review` |

The report exposes every L0 fixed-step event. The expected sequence is:

```text
validate_parameters -> compile_intent -> preflight -> approval -> revalidate
-> execute -> verify -> compensate (conditional) -> audit
```

The L1 Skill is allowed to interpret the problem, collect context, select a
candidate workflow and decide which L0 capability should be requested. It is
not allowed to issue an untyped write. The L0 Skill owns the deterministic
effect boundary: strict parameters, fixed control flow, approval binding,
one-shot dispatch, typed result verification, terminal state and audit.

## How to review the output

For each item in `writes`, verify all of the following:

- `l0_skill_id` is the expected versioned contract, not merely a tool name.
- `terminal_state` is `verified_success`; a successful adapter response alone
  is not sufficient.
- `evidence[].passed` is true and its source is an independent read contract.
- `l0_events` contains the fixed steps in contract order.
- `audit.ok` is true, proving the stored event chain is internally consistent.

At the business layer, `guarantees_review.problem_resolved` must be true,
`unbound_writes` and `unverified_successes` must both be zero, and the final LAN
and DC facts must independently show the desired state.

## Failure behavior and limits

The same L0 runtime fails closed when intent is incomplete or ambiguous, the
L0 Skill ID does not match the tool, the workflow prerequisites are absent,
approval does not match the immutable plan, preflight state changes, the nonce
is reused, verification fails or the event journal is tampered with. For a
contract that declares compensation, verification failure transitions through
the compensation path and verifies the rollback result; otherwise it stops in
a terminal failure state for operator handling.

This example deliberately does not claim that L1 model reasoning is 100%
accurate. It makes the L1 choices deterministic so the safety boundary can be
reproduced. It also uses a local profile handoff in place of live A2A transport
and an explicit simulated authorization in place of a DSH UI click. Those
transport and UI paths are covered separately by the migration regression
suite; real-device qualification still requires a controlled lab environment.
