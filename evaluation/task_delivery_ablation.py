"""One frozen two-arm diagnostic of output mediation, NOT a DSH/Runtime benchmark.

Uses a previously captured wire request. No tools, writes, source scripts,
semantic gold, graphs or model-authored executable L0 are created/executed.
No retry, resume, automatic judge or score is provided. Each output is retained.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tarfile
import time

import httpx

from dsh_adapter.hybrid_session import fingerprint
from skill_authoring import compiler, delivery, reasoning_transport
from skill_authoring.artifacts import read_json, write_artifacts
from skill_authoring.contracts import budget, seal
from network_runtime.l0.structured_schema import validate_data

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "http://127.0.0.1:11434"


def digest(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair(captured):
    original = copy.deepcopy(captured["governedRequest"])
    wire = captured["wireRequest"]
    old = json.loads(original["inputs"]["delivery_contract"])
    if (old["profile"] != delivery.CHOICE_PROFILE or wire["model"] != compiler.MODEL
            or original["tools"] != [] or original["runtimeAuthorityGranted"] is not False
            or reasoning_transport.messages(original) != wire["messages"]
            or wire["format"] != delivery.response_schema(old)
            or wire["think"] is not False or wire["stream"] is not False
            or wire["options"] != {k: v for k, v in compiler.MODEL_CONFIG.items() if k != "think"}):
        raise ValueError("exact frozen tool-free 9B choice-protocol input required")
    suffix = "\n" + delivery.generation(old)
    if not original["instructions"].endswith(suffix):
        raise ValueError("cannot isolate original output instruction without changing other policy")
    task = delivery.compile_task(None, {"task": original["inputs"]["original_task"],
                                        "source_material": original["inputs"]["source_material"]})
    modified = copy.deepcopy(original)
    modified["inputs"]["delivery_contract"] = json.dumps(task, ensure_ascii=False)
    modified["outputSchema"] = delivery.response_schema(task)
    modified["instructions"] = original["instructions"][:-len(suffix)] + "\n" + delivery.generation(task)
    new_wire = {**copy.deepcopy(wire), "messages": reasoning_transport.messages(modified), "format": modified["outputSchema"]}
    # Original task/source/caller/observations/limitations are byte-identical.
    assert {k: v for k, v in original["inputs"].items() if k != "delivery_contract"} == {
        k: v for k, v in modified["inputs"].items() if k != "delivery_contract"}
    assert all(budget(w)["accepted"] for w in (wire, new_wire))
    return {"choice": {"wire": wire, "contract": old}, "task": {"wire": new_wire, "contract": task}}


def run(captured_path, output):
    captured_path, output = Path(captured_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("new directory required; never overwrite or retry a frozen pair")
    captured = read_json(captured_path)
    arms = pair(captured)
    sources = fingerprint()
    sources["evaluation/task_delivery_ablation.py"] = digest(Path(__file__))
    with httpx.Client(timeout=240, trust_env=False) as client:
        tags = client.get(ENDPOINT + "/api/tags")
        tags.raise_for_status()
        models = [m for m in tags.json()["models"] if m["name"] == compiler.MODEL]
        if len(models) != 1 or models[0].get("digest") != captured["modelArtifact"]:
            raise ValueError("same exact frozen model artifact required")
        write_artifacts(output, {"freeze.json": seal({"kind": "known_snapshot_output_mediation_diagnostic",
            "capturedPath": str(captured_path), "capturedDigest": digest(captured_path), "sourceFiles": sources,
            "modelArtifact": captured["modelArtifact"], "arms": arms, "order": ["choice", "task"],
            "maxModelCalls": 2, "maxSecondsPerCall": 240, "retries": 0,
            "primaryQuestion": "Does removing model-selected output mediation change omissions on identical frozen evidence?",
            "reviewDimensions": ["facts_and_scope", "uncertainty_not_root_cause", "useful_new_evidence", "no_redundant_reads", "requested_format"],
            "reviewAuthority": "developer_review_not_independent_gold", "formalStageExit": False,
            "runtimeExecuted": False, "dshExecuted": False, "freshSkills": 0,
            "limits": ["one known snapshot; no causal population estimate", "order/caching can affect latency",
                       "contract, output instruction and schema change together as one presentation mechanism"]})})
        with tarfile.open(output / "source.tar.gz", "x:gz") as archive:
            for path in sources:
                archive.add(ROOT / path, arcname=path, recursive=False)
        rows = []
        for name, arm in arms.items():
            folder = output / name
            write_artifacts(folder, {"request.json": arm["wire"]})
            row = {"arm": name, "status": "unknown", "inputTokens": None, "outputTokens": None,
                   "taskSuccess": None, "semanticApproval": False}
            began = time.monotonic()
            try:
                response = client.post(ENDPOINT + "/api/chat", json=arm["wire"])
                response.raise_for_status()
                envelope = response.json()
                write_artifacts(folder / "response", {"envelope.json": envelope})
                row.update(inputTokens=envelope.get("prompt_eval_count"), outputTokens=envelope.get("eval_count"))
                if envelope.get("model") != compiler.MODEL or envelope.get("done") is not True or envelope.get("done_reason") != "stop":
                    raise ValueError("wrong model or incomplete reply")
                candidate = validate_data(arm["wire"]["format"], delivery.decode_response(envelope["message"]["content"]))
                rendered = delivery.render(arm["contract"], candidate)
                write_artifacts(folder / "delivery", {"report.json": rendered})
                row["status"] = "schema_valid_candidate_not_semantic_proof"
            except Exception as error:
                row.update(status="failed_or_unknown_no_retry", errorType=type(error).__name__, error=str(error)[:300])
            row["latencyMs"] = (time.monotonic() - began) * 1000
            write_artifacts(folder / "cost", {"report.json": seal(row)})
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = seal({"rows": rows, "semanticScores": None, "formalStageExit": False,
        "runtimeExecuted": False, "dshExecuted": False,
        "artifactDigests": {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()}})
    write_artifacts(output / "summary", {"report.json": report})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captured_request")
    parser.add_argument("new_output_directory")
    args = parser.parse_args()
    run(args.captured_request, args.new_output_directory)
