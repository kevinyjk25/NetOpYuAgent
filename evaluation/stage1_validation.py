"""Developer synthetic behavior checks on an explicitly reviewed generated Tree.

These expected traces never enter authoring input. No model, public network,
source scripts or Effect provider is called by this validator.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from pathlib import Path

from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import read_json, write_artifacts
from evaluation.structured_flow_demo import context
from evaluation.structured_flow_tree import StructuredFlowTree, compile_structured_tree
from network_runtime.capabilities import CapabilityContract
from network_runtime.contracts import sha256_json
from network_runtime.l0.flow import HostFlowConsent, parse_flow, run_read_flow
from network_runtime.l0.read_execution import HostReadBinding
from network_runtime.l0.structured_reads import parse_read_contract


def cases(name):
    if name == "wiring":
        args = {"device": {"id": "lab-sw1"}}
        first = ("get_interfaces", args)
        second = ("get_interface_counters", {"device": args["device"], "interface": {"name": "eth0"}})
        base = {"get_interfaces": {"interfaces": [{"name": "eth0", "adminUp": False}]},
                "get_interface_counters": {"counters": {"errors": 7}}}
        variants = [
            ("empty", {"get_interfaces": {"interfaces": []}}, "read_path_completed", [first]),
            ("up", {"get_interfaces": {"interfaces": [{"name": "eth0", "adminUp": True}]}}, "read_path_completed", [first]),
            ("zero_errors", {"get_interface_counters": {"counters": {"errors": 0}}}, "read_path_completed", [first, second]),
            ("errors_handoff", {}, "needs_l1", [first, second]),
            ("invalid_admin", {"get_interfaces": {"interfaces": [{"name": "eth0", "adminUp": "false"}]}}, "blocked", [first]),
            ("invalid_counter", {"get_interface_counters": {"counters": {"errors": "many"}}}, "blocked", [first, second]),
        ]
    elif name == "approval":
        args = {"changeId": "change-41"}
        first = ("get_change_request", args)
        second = ("get_device_health", {"deviceId": "lab-sw1"})
        base = {"get_change_request": {"approved": True, "device": {"id": "lab-sw1"}},
                "get_device_health": {"health": {"available": False}}}
        variants = [
            ("not_approved", {"get_change_request": {"approved": False, "device": {"id": "lab-sw1"}}}, "read_path_completed", [first]),
            ("available", {"get_device_health": {"health": {"available": True}}}, "read_path_completed", [first, second]),
            ("unavailable", {}, "needs_l1", [first, second]),
            ("invalid_approval", {"get_change_request": {"approved": "yes", "device": {"id": "lab-sw1"}}}, "blocked", [first]),
            ("invalid_health", {"get_device_health": {"health": {"available": "true"}}}, "blocked", [first, second]),
        ]
    elif name == "reference":
        args = {"serviceId": "service-17"}
        first = ("lookup_service", args)
        second = ("get_device_health", {"deviceId": "lab-sw1"})
        third = ("get_alarm", {"alarmId": "alarm-8"})
        base = {"lookup_service": {"ready": True, "device": {"id": "lab-sw1"}},
                "get_device_health": {"health": {"available": False, "alarmId": "alarm-8"}},
                "get_alarm": {"code": "LOS", "details": "synthetic raw details; do not publish"}}
        variants = [
            ("not_ready", {"lookup_service": {"ready": False, "device": {"id": "lab-sw1"}}}, "read_path_completed", [first]),
            ("available", {"get_device_health": {"health": {"available": True, "alarmId": "alarm-8"}}}, "read_path_completed", [first, second]),
            ("alarm_handoff", {}, "needs_l1", [first, second, third]),
            ("invalid_ready", {"lookup_service": {"ready": "true", "device": {"id": "lab-sw1"}}}, "blocked", [first]),
            ("invalid_alarm", {"get_alarm": {"code": 17, "details": "synthetic"}}, "blocked", [first, second, third]),
        ]
    else:
        raise ValueError("unknown stage-1 behavior fixture")
    return args, base, variants


def validate(case, run, reviewed_tree_digest):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    compilations = sorted(run.glob("round-*/compilation.json"))
    if len(compilations) != 1:
        raise ValueError("exactly one compiled generated candidate is required")
    compiled = read_json(compilations[0])
    if reviewed_tree_digest != compiled["treeDigest"]:
        raise ValueError("explicit reviewed Tree digest does not match generated candidate")
    original = read_json(compilations[0].with_name("tree.json"))
    if sha256_json(original) != reviewed_tree_digest:
        raise ValueError("generated Tree digest drift")
    original_reads = {k: parse_read_contract(r) for k, r in manifest["inputs"]["reads"].items()}
    if compiled != compile_structured_tree(manifest["inputs"]["bundle"], StructuredFlowTree.model_validate(original), original_reads, {}):
        raise ValueError("generated compilation differs from reviewed Tree and original contracts")
    proposal = parse_flow(compiled["flow"])
    reads = {c.contract_hash: c for c in original_reads.values()}
    inputs, base, variants = cases(case)
    ctx = replace(context(), scopes=frozenset({"network:read", "device_id:lab-sw1", "interface_name:eth0",
        "change_id:change-41", "service_id:service-17", "alarm_id:alarm-8"}))
    rows = []
    def one(name, outputs, expected, expected_calls, access=ctx, wrong_consent=False):
        calls, bindings = [], {}
        for digest, contract in reads.items():
            tool = contract.spec.tool
            def observe(args, tool=tool):
                calls.append((tool, args))
                return copy.deepcopy(outputs[tool]), {"source": "local_synthetic_stage1"}
            capability = CapabilityContract.from_metadata(tool, {"capability_id": contract.spec.capability,
                "action_type": "read_only", "domain": "network",
                "input_schema_digest": sha256_json(contract.spec.input_schema),
                "output_schema_digest": sha256_json(contract.spec.output_schema), "required_roles": ["netops"],
                "sensitivity": "internal", "scope_fields": list(contract.spec.resource_scopes), "freshness_limit_seconds": 10},
                source="local_fixture")
            bindings[digest] = HostReadBinding(digest, capability, frozenset({"network:read"}), observe)
        try:
            result = run_read_flow(proposal, inputs, reads=reads, effects={}, bindings=bindings, context=access,
                consent=HostFlowConsent("wrong" if wrong_consent else compiled["qualification"]["flowDigest"], sha256_json(inputs)))
            actual = result["status"]
            assert not result["effectExecuted"]
        except PermissionError:
            if not wrong_consent:
                raise
            actual, result = "PermissionError", {"beforeAnyHostCall": True}
        rows.append({"case": name, "expectedStatus": expected, "status": actual,
            "expectedCalls": expected_calls, "calls": calls, "passed": actual == expected and calls == expected_calls,
            "result": result})
    for name, patch, outcome, expected_calls in variants:
        one(name, {**base, **patch}, outcome, expected_calls)
    one("unauthenticated", base, "blocked", [], access=replace(ctx, authenticated=False))
    one("missing_scope", base, "blocked", [], access=replace(ctx, scopes=frozenset()))
    one("wrong_consent", base, "PermissionError", [], wrong_consent=True)
    return seal({"caseFamily": case, "treeDigest": reviewed_tree_digest, "manifestDigest": manifest["reportDigest"],
        "reviewerKind": "explicit_developer_review_required_not_independent_gold", "cases": rows,
        "passed": sum(r["passed"] for r in rows), "failed": sum(not r["passed"] for r in rows),
        "modelCalls": 0, "networkCalls": 0, "sourceScriptCalls": 0, "effectCalls": 0,
        "runtimeAuthorityGranted": False, "wholeSkillTranslationProven": False,
        "largeRuntimeAB": False, "evidenceRole": "known_synthetic_generated_read_region_validation"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["wiring", "approval", "reference"])
    parser.add_argument("run")
    parser.add_argument("--reviewed-tree-digest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = validate(args.case, args.run, args.reviewed_tree_digest)
    write_artifacts(args.output, {"report.json": report})
    print(report["passed"], "passed;", report["failed"], "failed")
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
