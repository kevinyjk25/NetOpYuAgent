"""Three disclosed host-mapped result probes on known Stage 2 read prefixes.

This does NOT rerun or score translation. Read plans/tasks/fixtures are retained;
only the candidate protocol and host result checks change. No script execution,
effect calls, automatic activation, new cohort or DSH A/B. Output must be new.
"""
from __future__ import annotations

import argparse
import copy
import json
import tarfile
from pathlib import Path

from evaluation.flow_checkpoint import implementation
from evaluation.hybrid_authoring import compile_proposal
from evaluation.hybrid_live_demo import run
from evaluation.hybrid_result_review import prepare_binding
from evaluation.structured_binding_probe import read_json, write_artifacts
from network_runtime.contracts import sha256_json

CASES = {
    "notion": ("/markdown", "observed-page", "Return the observed page Markdown without inventing content.", []),
    "handoff": ("/text", "observed-notes", "Preserve the outgoing notes as observed, not as independently attested facts.", [
        ("handoff-draft", "Prepare a source-supported asynchronous handoff draft without fabricated facts or missing duties.", "unverified_reasoning")]),
    "documentation": ("/entries", "observed-entries", "Preserve the observed directory entries without inferring configuration.", [
        ("inspect-code", "Inspect configuration and code before claiming project-specific installation or test commands.", "missing_observation"),
        ("readme-draft", "Prepare an appropriate README supported by actual codebase observations.", "unverified_reasoning")]),
}


def seal(body):
    return {**body, "reportDigest": sha256_json(body)}


def _semantic_flow(flow):
    """Compare ONLY the two known JSON-as-text envelopes independent of key order.

    No other instruction, argument, tool, branch, model configuration or limit
    may change during this zero-model recompilation. This is not a hash bypass.
    """
    flow = copy.deepcopy(flow)
    for node in flow["nodes"]:
        if node["kind"] == "reason":
            for name in ("source_material", "authoring_boundaries"):
                literal = node["inputs"]["fields"].get(name)
                if literal is not None and literal["kind"] == "literal":
                    literal["value"] = json.loads(literal["value"])
    return flow


def prepare(stage2_root, output):
    stage2_root, output = Path(stage2_root).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(stage2_root):
        raise FileExistsError("new output outside frozen Stage 2 evidence required")
    prepared = {}
    for case, (pointer, field, statement, open_duties) in CASES.items():
        packet = read_json(stage2_root / f"hybrid-v7-preparation/{case}/author-input.json")
        original = read_json(stage2_root / f"hybrid-v7-run/{case}/round-00/compilation.json")
        if original != seal({k: v for k, v in original.items() if k != "reportDigest"}):
            raise ValueError("original compilation digest drift")
        compilation = compile_proposal(packet, original["suppliedPages"], original["plan"])
        if _semantic_flow(compilation["flow"]) != _semantic_flow(original["flow"]):
            raise ValueError("zero-call canonical recompilation changed flow semantics")
        duties = [{"id": "read-source", "kind": "read_completed", "statement": "Read the caller-selected source through its declared host tool.",
                   "source_ref": "task", "region": "n0", "read_node": "read"},
                  {"id": "preserve-observation", "kind": "observed_value", "statement": statement, "source_ref": "task",
                   "region": "n0", "read_node": "read", "pointer": pointer, "field": field}]
        duties.extend({"id": key, "kind": "open_semantics", "statement": text, "source_ref": "task", "reason": reason}
                      for key, text, reason in open_duties)
        span = {"path": "task", "start": 0, "end": len(packet["task"]), "quote": packet["task"]}
        mapping = {"sourceDigest": packet["bundle"]["bundleDigest"], "taskDigest": sha256_json(packet["task"]),
            "reviewKind": "developer_ai_not_independent_gold",
            "duties": [{**{k: d[k] for k in ("id", "statement", "source_ref")}, "source": span} for d in duties]}
        contract = {"api_version": "netopyu.io/hybrid-result/v1", "source_digest": mapping["sourceDigest"],
            "task_digest": mapping["taskDigest"], "mapping_digest": sha256_json(mapping), "candidate_node": "n7", "duties": duties}
        _, _, qualified = prepare_binding(packet, compilation, contract, mapping)
        review = {"case": case, "compilationDigest": compilation["reportDigest"], "resultContractDigest": qualified["contractDigest"],
            "decision": "admit_local_read_reason_only", "reviewKind": "developer_ai_not_independent_gold",
            "rationale": "Known development source and task retained. Host-authored exact observation projection only; "
                         "all open drafting duties stay unverified. No new translation accuracy or task completion claim."}
        prepared[case] = {"packet.json": packet, "compilation.json": compilation, "stage2-compilation.json": original, "mapping.json": mapping,
                          "contract.json": contract, "review.json": review}
    for case, files in prepared.items():
        write_artifacts(output / case / "inputs", files)
    manifest = seal({"cases": list(CASES), "inputDigests": {c: {name: sha256_json(v) for name, v in files.items()}
                        for c, files in prepared.items()},
        "implementation": implementation("evaluation/hybrid_result_batch.py", "evaluation/hybrid_live_demo.py",
            "evaluation/hybrid_result_review.py", "evaluation/hybrid_reasoning_transport.py", "evaluation/hybrid_behavior.py",
            "evaluation/hybrid_authoring.py"),
        "hostAuthoredResultMappings": True, "translationCalls": 0, "expectedResultValuesProvided": False,
        "canonicalRecompilation": "v8; source/annotation JSON key order only, original flow and all source strings retained",
        "wholeTaskDutyCoverageProven": False, "evidenceRole": "known_development_result_wiring_not_generalization"})
    write_artifacts(output / "freeze", {"manifest.json": manifest})
    with tarfile.open(output / "freeze/source-snapshot.tar.gz", "x:gz") as archive:
        repository = Path(__file__).resolve().parents[1]
        for relative in manifest["implementation"]:
            archive.add(repository / relative, arcname=relative, recursive=False)
    return manifest


def execute(stage2_root, output, *, max_model_calls=0):
    if type(max_model_calls) is not int or max_model_calls not in (0, len(CASES)):
        raise ValueError("use zero for preparation, or exactly three one-attempt local model probes")
    manifest = prepare(stage2_root, output)
    if max_model_calls == 0:
        return manifest
    rows = []
    for case in CASES:
        root = Path(output) / case
        inputs = root / "inputs"
        report = run(inputs / "packet.json", inputs / "compilation.json", inputs / "review.json", root / "execution",
            case=case, max_model_calls=1, result_contract_path=inputs / "contract.json", result_mapping_path=inputs / "mapping.json")
        assessment = report["execution"].get("resultAssessment", {})
        rows.append({"case": case, "reportDigest": report["reportDigest"], "graphStatus": report["execution"]["status"],
                     "resultAssessment": assessment, "modelCalls": report["modelCalls"], "latencyMs": report["runtimeWallLatencyMs"]})
    report = seal({"manifestDigest": manifest["reportDigest"], "rows": rows, "actualModelCallCount": sum(len(r["modelCalls"]) for r in rows),
                   "automaticTranslationAccuracy": None, "wholeSkillAccuracy": None, "semanticDraftAccuracy": None,
                   "largeRuntimeABUnlocked": False, "status": "result_boundary_probe_completed_not_semantic_generalization"})
    write_artifacts(Path(output) / "summary", {"report.json": report})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage2_root")
    parser.add_argument("output")
    parser.add_argument("--max-model-calls", type=int, default=0)
    a = parser.parse_args()
    report = execute(a.stage2_root, a.output, max_model_calls=a.max_model_calls)
    print(json.dumps({"reportDigest": report["reportDigest"], "output": a.output,
        "rows": [{"case": r["case"], "graphStatus": r["graphStatus"], "resultStatus": r["resultAssessment"].get("status")}
                 for r in report.get("rows", [])]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
