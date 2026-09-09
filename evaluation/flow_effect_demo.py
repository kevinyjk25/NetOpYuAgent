"""Synthetic business condition + real local read + existing mock Effect engine."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import tempfile
from pathlib import Path

from evaluation.flow_local_demo import build_local_flow
from evaluation.read_local_demo import DATASET, host_binding
from network_provider.local_inventory import LocalInventoryReader
from network_runtime.access import ObservationAccessContext
from network_runtime.capabilities import DataSensitivity
from network_runtime.contracts import sha256_json
from network_runtime.engine import NetworkRuntime
from network_runtime.l0.flow import EffectTarget, FlowProposal, HostFlowConsent, qualify_flow
from network_runtime.l0.flow_admission import HostFlowGate
from network_runtime.l0.models import ReadObjectSchema
from network_runtime.l0_skills import REGISTRY


ARGUMENTS = {"user_id": "erin", "reason": "Local flow/transaction wiring test"}


def local_gate(dataset: Path) -> HostFlowGate:
    """The site-to-access rule is a wiring fixture, NOT a sourced business rule."""
    proposal, reads = build_local_flow()
    skill = REGISTRY.for_tool("lan", "grant_user_access")
    raw = proposal.model_dump()
    raw["nodes"] = list(raw["nodes"])
    raw["source_digest"] = sha256_json({"fixture": "campus snapshot gates local simulated access", "read_source": proposal.source_digest})
    raw["purpose"] = "Synthetic branch-to-Effect wiring; not a production access policy."
    raw["nodes"][3] = {"kind": "effect_candidate", "id": "done", "binding_id": "grant",
                       "arguments": {key: {"kind": "constant", "value": value} for key, value in ARGUMENTS.items()}}
    proposal = FlowProposal.model_validate(raw)
    schema = ReadObjectSchema.model_validate({"type": "object", "additionalProperties": False,
        "properties": {key: {"type": value.type} for key, value in skill.compiled_contract.spec.parameters.items()},
        "required": [key for key, value in skill.compiled_contract.spec.parameters.items() if value.required]})
    target = EffectTarget(profile="lan", tool=skill.tool_name, skill_id=skill.skill_id,
                          contract_hash=skill.contract_hash, input_schema=schema)
    effects = {"grant": target}
    arguments = {"device_id": "campus-sw1"}
    packet = qualify_flow(proposal, reads, effects)
    return HostFlowGate(proposal, arguments, reads, effects,
                        {key: host_binding(contract, LocalInventoryReader(dataset)) for key, contract in reads.items()},
                        ObservationAccessContext(subject_id="local-flow-host", roles=frozenset({"network-reader"}),
                            scopes=frozenset({"inventory:read", "device_id:campus-sw1"}),
                            purpose="Local flow wiring", clearance=DataSensitivity.INTERNAL),
                        HostFlowConsent(packet["flowDigest"], sha256_json(arguments)))


async def run_demo(*, approve_local_simulation: bool) -> dict:
    if not approve_local_simulation:
        raise PermissionError("explicit local simulation approval required")
    from profiles.lan import tools as lan_tools
    saved_changes = copy.deepcopy(lan_tools._LAN_ACCESS_CHANGES)
    saved_operations = copy.deepcopy(lan_tools._MOCK_OPERATION_STATE)
    names = ("NETOPYU_DSH_BACKEND", "NETOPYU_DSH_NETWORK_RUNTIME_STORE", "NETOPYU_DSH_TOOL_RESULT_STORE")
    previous = {key: os.environ.get(key) for key in names}
    outcomes = []
    try:
        with tempfile.TemporaryDirectory(prefix="netopyu-flow-effect-") as directory:
            root = Path(directory)
            dataset = root / "inventory.json"
            dataset.write_bytes(DATASET.read_bytes())
            for case in ("success", "branch_drift", "rollback"):
                lan_tools._LAN_ACCESS_CHANGES.clear()
                lan_tools._MOCK_OPERATION_STATE.clear()
                dataset.write_bytes(DATASET.read_bytes())
                journal = root / (case + ".sqlite")
                os.environ.update(NETOPYU_DSH_BACKEND="mock", NETOPYU_DSH_NETWORK_RUNTIME_STORE=str(journal),
                                  NETOPYU_DSH_TOOL_RESULT_STORE=str(root / "results.sqlite"))

                def fault(stage, _plan):
                    if case == "rollback" and stage == "before_verify":
                        lan_tools._LAN_ACCESS_CHANGES.append({"user_id": "erin", "op": "fault",
                            "changes": {"radius": "fail", "dot1x": "rejected", "nac": "quarantine", "vlan": None},
                            "reason": "isolated verification failure"})

                runtime = NetworkRuntime(journal, flow_gates={"local": local_gate(dataset)}, fault_hook=fault)
                target = runtime.flow_gates["local"].effects["grant"]
                prepared = await runtime.prepare("lan", target.tool, dict(ARGUMENTS), l0_skill_id=target.skill_id, flow_gate_id="local")
                if prepared["status"] != "plan_ready":
                    raise RuntimeError("local flow admission did not prepare")
                plan = prepared["plan"]
                if case == "branch_drift":
                    data = json.loads(dataset.read_text())
                    data["campus-sw1"]["site"] = "idc"
                    dataset.write_text(json.dumps(data))
                outcome = await runtime.execute(plan_id=plan["plan_id"], plan_hash=plan["plan_hash"],
                    execution_nonce=prepared["execution_nonce"], allow_destructive=True,
                    approval_request_id="local-flow-" + case, approval_actor="local-demo-operator")
                audit = runtime.inspect(plan["plan_id"])
                outcomes.append({"case": case, "state": outcome.state.value, "ok": outcome.ok,
                    "planHash": plan["plan_hash"], "flowBinding": plan["flow_binding"],
                    "graph": audit["graph_execution"], "mockChanges": copy.deepcopy(lan_tools._LAN_ACCESS_CHANGES)})
        body = {"evidenceRole": "synthetic_local_wiring_not_translation", "modelCalls": 0,
                "networkBackend": "mock", "sourceReads": "actual temporary local inventory file",
                "outcomes": outcomes, "productionCorrectnessProven": False}
        return {**body, "reportDigest": sha256_json(body)}
    finally:
        lan_tools._LAN_ACCESS_CHANGES[:] = saved_changes
        lan_tools._MOCK_OPERATION_STATE.clear()
        lan_tools._MOCK_OPERATION_STATE.update(saved_operations)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approve-local-simulation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(asyncio.run(run_demo(approve_local_simulation=args.approve_local_simulation)),
                                ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
