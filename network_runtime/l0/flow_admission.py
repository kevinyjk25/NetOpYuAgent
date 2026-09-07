"""Host-only business-flow admission for the existing Effect engine.

Re-evaluate reads rather than accepting a caller's flow report. This local
prototype does not load providers from a plan, attest source freshness, or
make observations and an external write globally atomic.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from network_runtime.access import ObservationAccessContext
from network_runtime.contracts import sha256_json

from .flow import EffectTarget, FlowProposal, HostFlowConsent, qualify_flow, run_read_flow
from .models import CompiledAtomicRead, StrictModel
from .read_execution import HostReadBinding


class FlowAdmission(StrictModel):
    gate_id: str
    flow_digest: str
    arguments_digest: str
    context_digest: str
    decision: dict[str, Any]
    decision_digest: str
    read_report_digest: str


def validate_admission(raw: dict, *, profile: str, tool: str, skill_id: str,
                       contract_hash: str, arguments: dict) -> FlowAdmission:
    admission = FlowAdmission.model_validate(raw)
    if sha256_json(admission.decision) != admission.decision_digest:
        raise ValueError("flow decision digest mismatch")
    candidate = admission.decision["candidate"]
    target = candidate["target"]
    if (target["profile"], target["tool"], target["skill_id"], target["contract_hash"]) != (
        profile, tool, skill_id, contract_hash,
    ) or sha256_json(candidate["arguments"]) != sha256_json(arguments):
        raise ValueError("flow candidate differs from exact Effect target/arguments")
    return admission


@dataclass(frozen=True)
class HostFlowGate:
    proposal: FlowProposal
    arguments: dict[str, Any]
    reads: Mapping[str, CompiledAtomicRead]
    effects: Mapping[str, EffectTarget]
    bindings: Mapping[str, HostReadBinding]
    context: ObservationAccessContext
    consent: HostFlowConsent

    def evaluate(self, gate_id: str) -> tuple[dict, float]:
        started = time.monotonic()
        packet = qualify_flow(self.proposal, self.reads, self.effects)
        report = run_read_flow(self.proposal, self.arguments, reads=self.reads, effects=self.effects,
                               bindings=self.bindings, context=self.context, consent=self.consent)
        if report["status"] != "awaiting_effect_admission":
            raise ValueError("business flow did not reach an admissible Effect candidate")
        # Keep values and provider provenance, not per-run receipt timestamps.
        # Full read payload comparison is deliberately conservative.
        steps = []
        for row in report["trace"]:
            if row["kind"] == "read":
                receipt = row["receipt"]
                steps.append({"node": row["node"], "kind": "read", "payload": receipt["payload"],
                              "contractHash": receipt["contractHash"], "requestDigest": receipt["requestDigest"],
                              "providerEvidence": receipt["providerEvidence"]})
            else:
                steps.append(row)
        decision = {"candidate": report["candidate"], "steps": steps}
        context = asdict(self.context)
        context["roles"], context["scopes"] = sorted(self.context.roles), sorted(self.context.scopes)
        value = FlowAdmission(gate_id=gate_id, flow_digest=packet["flowDigest"],
                              arguments_digest=report["argumentsDigest"], context_digest=sha256_json(context),
                              decision=decision, decision_digest=sha256_json(decision),
                              read_report_digest=report["reportDigest"]).model_dump(mode="json")
        # Count the entire read path including provider latency, not only the
        # final read's completion. Check this again immediately before dispatch.
        deadline = started + self.proposal.max_read_age_seconds
        if time.monotonic() > deadline:
            raise ValueError("business-flow read budget expired")
        return json.loads(json.dumps(value, allow_nan=False)), deadline

    def revalidate(self, persisted: dict) -> tuple[dict, float]:
        previous = FlowAdmission.model_validate(persisted)
        current, deadline = self.evaluate(previous.gate_id)
        if any(current[key] != persisted[key] for key in (
            "gate_id", "flow_digest", "arguments_digest", "context_digest", "decision_digest", "decision",
        )):
            raise ValueError("approved business-flow evidence/route changed")
        return current, deadline
