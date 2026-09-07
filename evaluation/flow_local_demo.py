"""Hand-authored local business-flow smoke; no LLM or public-Skill score."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.read_local_demo import DATASET, ReadIntentDraft, build_forward_proposal, host_binding, source_bundle
from network_provider.local_inventory import LocalInventoryReader
from network_runtime.access import ObservationAccessContext
from network_runtime.capabilities import DataSensitivity
from network_runtime.contracts import sha256_json
from network_runtime.l0.compiler import compile_documents
from network_runtime.l0.flow import FlowProposal, HostFlowConsent, qualify_flow, run_read_flow


SOURCE = Path(__file__).resolve().parents[1] / "examples/read-flow/flow-source.md"


def build_local_flow() -> tuple[FlowProposal, dict]:
    """Explicit development fixture; never count this as model translation."""
    read = build_forward_proposal(source_bundle(), ReadIntentDraft(
        purpose="Read a local inventory snapshot, not live network health.",
        tool="read_inventory_device", action_type="read_only", unresolved_questions=(),
    ))
    contract = compile_documents([read.to_manifest()])[0]

    def ref(source, field):
        return {"kind": "reference", "source": source, "field": field}

    proposal = FlowProposal.model_validate({
        "api_version": "netopyu.io/l0-flow-proposal/v1",
        "source_digest": "sha256:" + hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "purpose": "Route a local inventory read by site; pause for reasoning outside campus.",
        "input_schema": contract.spec.input_schema.model_dump(by_alias=True),
        "entry": "lookup", "max_read_age_seconds": 30,
        "nodes": [
            {"id": "lookup", "kind": "read", "contract_hash": contract.contract_hash,
             "arguments": {"device_id": ref("input", "device_id")}, "next": "site"},
            {"id": "site", "kind": "branch", "left": ref("lookup", "site"),
             "equals": {"kind": "constant", "value": "campus"}, "on_true": "reread", "on_false": "reason"},
            {"id": "reread", "kind": "read", "contract_hash": contract.contract_hash,
             "arguments": {"device_id": ref("lookup", "device_id")}, "next": "done"},
            {"id": "done", "kind": "end", "outcome": "read_path_completed",
             "explanation": "Local read path complete; no live-health or whole-Skill correctness claim."},
            {"id": "reason", "kind": "end", "outcome": "needs_l1",
             "explanation": "Host should request L1 reasoning. No model is called by this smoke."},
        ],
    })
    return proposal, {contract.contract_hash: contract}


def run_demo(*, allow_local_read: bool) -> dict:
    if not allow_local_read:
        raise PermissionError("explicit --allow-local-read required")
    proposal, reads = build_local_flow()
    packet = qualify_flow(proposal, reads, {})
    bindings = {key: host_binding(contract, LocalInventoryReader(DATASET)) for key, contract in reads.items()}
    before = DATASET.read_bytes()
    outcomes = []
    for device in ("campus-sw1", "idc-sw1"):
        # Separate host-scoped requests, not a wildcard or an LLM-supplied role.
        context = ObservationAccessContext(
            subject_id="local-flow-demo", roles=frozenset({"network-reader"}),
            scopes=frozenset({"inventory:read", "device_id:" + device}),
            purpose="Explicitly authorized local inventory demonstration", clearance=DataSensitivity.INTERNAL,
        )
        arguments = {"device_id": device}
        outcomes.append(run_read_flow(proposal, arguments, reads=reads, effects={}, bindings=bindings,
                                      context=context, consent=HostFlowConsent(packet["flowDigest"], sha256_json(arguments))))
    body = {
        "evidenceRole": "hand_authored_local_wiring_not_translation",
        "modelCalls": 0, "thirdPartyCodeExecuted": False,
        "sourceDigest": proposal.source_digest, "flowDigest": packet["flowDigest"],
        "datasetUnchanged": before == DATASET.read_bytes(),
        "datasetDigest": "sha256:" + hashlib.sha256(before).hexdigest(), "outcomes": outcomes,
    }
    return {**body, "reportDigest": sha256_json(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-local-read", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Reserve output before invoking even read tools. Never overwrite prior evidence.
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(run_demo(allow_local_read=args.allow_local_read), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
