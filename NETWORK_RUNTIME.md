# NetOpYu L1 Network Runtime

## Implementation status

P0.5 and the P1 Runtime Foundation are implemented for the supported local
deployment. `network_runtime/` owns strict compilation, inventory resolution,
immutable schema-v4 plans, first-class Network L0 Skill and IntentSpec contracts,
plan hashing, TTL, target locks, one-shot nonce consumption, workflow ordering,
approval-time and execution-time preflight comparison, risk summaries,
versioned typed verification, versioned compensation, crash reconciliation and
the hash-chained SQLite audit journal. DSH approval and Tool Guard are both
bound to the exact plan hash before `runtime-execute` is allowed.

Within the network domain, a **Network L0 Skill** is a deterministic execution
contract with no model calls inside its step graph. A **Network L1 Skill** is
DSH model guidance for generalized diagnosis and business scenarios. L1 Skills
may select and supply candidate inputs to L0 Skills, but cannot alter their
steps, verification, compensation or success decision. This terminology is
separate from the broader architecture shorthand in which DSH is the agent L0
framework and NetOpYu is its domain layer.

All registered mock writes have reviewed contracts and typed verifier reads.
The pragmatic backend currently permits only `edit_device_config`, whose
snapshot/apply/verify/rollback behavior is reviewed. Other pragmatic,
MCP/OpenAPI and remote-peer writes fail closed until a same-source L1 contract
is added; a local mock contract can never authorize an external same-name tool.

## Boundary

DSH is L0: conversation, model invocation, sessions, UI, approvals, skills and
subagent lifecycle. NetOpYu's L1 Network Runtime must own every network-domain
guarantee between an LLM proposal and a device, controller, MCP or OpenAPI
side effect.

An LLM may propose intent and candidate arguments. It must never directly
select an unregistered executor, relax validation, mark an operation verified,
or decide that rollback is unnecessary.

## Deterministic execution pipeline

Every mutating skill must pass the same state machine:

1. **Resolve capability** — select one versioned skill and reject ambiguous or
   unsupported intent.
2. **Compile parameters** — extract into a strict typed schema; normalize
   device, interface, site and service identifiers from inventory.
3. **Validate semantics** — enforce required fields, enums, ranges, dependency
   rules, target existence, authorization scope and current-state preconditions.
4. **Build plan** — emit immutable ordered steps, expected state transitions,
   verification predicates and rollback steps. Hash the canonical plan.
5. **Preflight** — gather fresh state, establish reachability, take snapshots,
   calculate diff/blast radius and reject stale or conflicting state.
6. **Approve** — bind the DSH allowed-once decision and Tool Guard grant to the
   exact plan hash, target set, arguments and expiry.
7. **Execute** — use an idempotency key and per-target state machine; never
   retry an unknown write outcome without reconciliation.
8. **Verify** — evaluate machine-readable postconditions using independent
   reads, not the write command's success message.
9. **Commit or compensate** — commit only after every required predicate passes;
   otherwise run bounded rollback and verify the restored state.
10. **Record evidence** — persist the redacted plan, snapshots, diff, approval,
    observations, predicate results, rollback outcome and terminal status.

## Core contracts

The runtime exposes versioned JSON contracts:

- `PreparedPlan`: normalized arguments and provenance, resolved targets, risk,
  typed preflight, L0 Skill/Intent hashes, deterministic step contract,
  workflow binding, verifier/rollback contract, TTL and hash.
- `IntentSpec`: normalized operation, exact resolved targets, desired state,
  constraints, argument digest and provenance. Missing or unresolved intent is
  clarified before a plan exists.
- `L0SkillContract`: versioned tool binding, target fields, fixed step graph,
  desired-state compiler and failure policy.
- `Evidence`: source, target, timestamp, typed facts/digest, predicate and
  expected value. Rendered command prose is never accepted as proof.
- `ExecutionOutcome`: result reference, all evidence, error/rollback result and
  terminal state.
- `NetworkJournal`: monotonic state transitions, approval identity, nonce
  consumption, resource locks, outcome and per-plan SHA-256-chained events.
- `WorkflowTemplate/WorkflowRun`: reviewed allowed tools, target-bound
  prerequisite observations, exact template hash and session lifecycle.

No free-form string may be used as proof that a postcondition passed.

## Accuracy and safety definition

"100% stable" cannot mean that every network command succeeds: devices,
controllers and links can fail. The enforceable guarantee is:

- zero unvalidated or unapproved writes;
- zero success reports without independent postcondition evidence;
- zero silent retries after an indeterminate write;
- zero rollback-success reports without restoration evidence;
- every non-success path ends fail-closed with a durable, actionable state.

These invariants must be exhaustive in the simulator and contract-test corpus.

## P0.5 implementation slices

### P0.5a — contracts and state machine

Implemented in `contracts.py`, `journal.py`, the DSH pre-execute hook and the
persistent Worker startup recovery path.

### P0.5b — parameter compiler

Implemented in `validation.py`. Unknown fields and types, unresolved entities,
forbidden configuration, invalid ranges/enums and missing safety arguments are
rejected before a plan or approval is created.

### P0.5c — skill workflow compiler

Implemented in `workflows.py`. Mutating skills without complete reviewed write
coverage prevent manifest startup. LAN/DC diagnostic writes require fresh,
same-target prerequisite observations; skipped/cross-skill writes are denied
before approval.

### P0.5c.1 — Network L0 Skill and intent compiler

Implemented in `l0_skills.py`. Every reviewed local write maps one-to-one to a
versioned deterministic L0 Skill. DSH receives the exact skill id from the
runtime manifest; missing or mismatched ids reject preparation. IntentSpec and
the L0 contract/step hashes are immutable parts of schema-v4 plans and approval
summaries. Missing fields produce `intent_incomplete`; unresolved target ids
produce `intent_target_ambiguous` and a deterministic clarification request.

### P0.5d — verification and rollback

Implemented for every registered local write contract. Unknown write outcomes
are never retried; device/simulator state is read to reconcile. Rollback is only
reported when a typed independent read exactly matches preflight. Otherwise the
plan retains its target lock and ends `manual_intervention_required`.

### P0.5e — deterministic network lab

The local contract lab covers malformed/ambiguous parameters, duplicate grants,
target conflicts, disconnect-before/after-effect, verification failure, exact
rollback success/failure, worker crash recovery, DSH rejection, partial batch
preparation and workflow step skipping. Real MCP/OpenAPI/A2A write contracts
remain a P1 integration task because this deployment intentionally has no real
network or external approval environment.

## P1 Runtime Foundation

The provider-facing foundation is implemented without pretending that the
local simulators are real controllers:

- every approved write is re-preflighted immediately before send; a changed
  typed snapshot ends as `precondition_changed` and the write count stays zero;
- verifier and compensator behavior is selected only through explicit
  registries, and missing handlers reject a plan before approval;
- DC verification uses the public `dc_get_applied_config` read contract instead
  of importing simulator private state;
- tool contracts bind the verifier and compensator ids into the immutable plan;
- every plan event includes `prev_event_hash` and `event_hash`; `runtime-audit`
  independently detects modified event state or payloads.

The remaining P1 integration work is provider-specific: implement real
MCP/OpenAPI/A2A write adapters, machine-readable snapshots, idempotency keys,
reconciliation reads and compensators for each external system. Until a source
has all of those contracts it continues to fail closed.

Automatic L1-to-L0 promotion is intentionally deferred. It requires production
business experience, representative trajectories and separately reviewed
qualification criteria. Offline mining continues to create uninstalled
proposals only; it never promotes executable L0 contracts.

## P0.5 release gate

- 100% of registered mutating skills use compiled schemas and plans.
- 100% of registered local writes bind one immutable L0 Skill and IntentSpec.
- Missing, unresolved or mismatched intent never creates an executable plan.
- LLM-visible write tools cannot bypass the L0 Skill step contract.
- 100% of write paths require plan-bound one-shot approval.
- 100% of terminal success states have independent verification evidence.
- Every write has an explicit idempotency/reconciliation policy.
- Every reversible write has tested compensation and restoration predicates.
- No direct pragmatic executor is callable outside the Network Runtime.
- Mutation, fault-injection and restart-recovery suites pass in CI.

Run the gate and inspect evidence locally:

```bash
scripts/netopyu-dsh retirement
scripts/netopyu-dsh runtime-list
scripts/netopyu-dsh runtime PLAN_ID
scripts/netopyu-dsh runtime-audit PLAN_ID
scripts/netopyu-dsh l0-skills
```
