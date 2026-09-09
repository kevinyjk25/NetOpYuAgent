"""Explicit-answer decoder canaries; never Skill accuracy or Runtime evidence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx
from jsonschema import Draft202012Validator

from evaluation.flow_translation import _write, local_sources
from evaluation.flow_translation_batch import fingerprint
from evaluation.flow_tree import FlowTree, compile_report
from evaluation.flow_tree_authoring import ROOT, digest_file, receipt, tree_request, verify_receipt
from evaluation.translation_case_authoring import OllamaAnchoredAuthorAdapter
from network_runtime.contracts import sha256_json

PROTOCOL = "explicit-tree-decoder-canary/v1"


def sources():
    return local_sources().model_copy(update={"source_text":
        "Protocol canary fixture: explicitly requested tree constructors, not source translation or live health.\n",
        "source_path": "synthetic-decoder-canary"})


def cases() -> list[dict]:
    def end(outcome):
        return {"kind": "end", "source_id": "s0001", "outcome": outcome}

    def branch(yes, no):
        return {"kind": "if_equal", "source_id": "s0001", "left": {
            "kind": "reference", "source": "input", "field": "device_id"},
            "equals": {"kind": "constant", "value": "campus-sw1"},
            "true_source_id": "s0001", "false_source_id": "s0001", "when_equal": yes, "otherwise": no}

    targets = {
        "end": {"steps": [end("read_path_completed")], "issues": []},
        "unsupported": {"steps": [end("unsupported")], "issues": [{"kind": "missing_host_capability",
            "source_id": "s0001", "question": "Canary declares a missing script capability; no execution is requested."}]},
        "conditional": {"steps": [branch([end("needs_l1")], [end("read_path_completed")])], "issues": []},
        "empty-branch": {"steps": [branch([], []), end("needs_l1")], "issues": []},
    }
    rows = []
    for name, value in targets.items():
        expected = {"business_source_ids": ["s0001"], **value}
        compile_report(sources(), FlowTree.model_validate(expected))
        for arm in ("format-only", "format-visible", "json-visible"):
            wire = tree_request(sources())
            wire["options"]["num_predict"] = 900
            schema = wire["format"]
            Draft202012Validator(schema).validate(expected)
            payload = {"expectedObject": expected, "task": "Copy expectedObject exactly as JSON. Preserve every key, value, list order and empty list. No commentary or added nodes."}
            if arm != "format-only":
                payload["outputSchema"] = schema
            wire["messages"] = [{"role": "system", "content": "This is a JSON protocol copying canary, not an agent or Skill task. Never execute anything. Return only the requested JSON object."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
            if arm == "json-visible":
                wire["format"] = "json"
            rows.append({"id": name + "--" + arm, "probe": name, "arm": arm,
                "wireRequest": wire, "expected": expected})
    return rows


def freeze(root: Path) -> dict:
    # No dependency on ignored historical artifacts: reproducible from a checkout.
    files = set(fingerprint()) | {"evaluation/flow_grounded_translation.py", "evaluation/flow_source_selection.py",
        "evaluation/flow_tree.py", "evaluation/flow_tree_authoring.py", "evaluation/flow_tree_canary.py"}
    body = {"protocol": PROTOCOL, "evidenceRole": "explicit_answer_protocol_probe_not_translation",
        "model": OllamaAnchoredAuthorAdapter().preflight(), "cases": cases(), "attemptsPerCase": 1,
        "implementation": {name: digest_file(ROOT / name) for name in sorted(files)}}
    manifest = {**body, "manifestDigest": sha256_json(body)}
    root.mkdir(parents=True, exist_ok=False)
    _write(root / "manifest.json", manifest)
    return manifest


def load(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest["manifestDigest"] != sha256_json({k: v for k, v in manifest.items() if k != "manifestDigest"})
            or manifest["protocol"] != PROTOCOL or manifest["cases"] != cases()
            or any(digest_file(ROOT / name) != digest for name, digest in manifest["implementation"].items())):
        raise ValueError("canary manifest/implementation drift")
    return manifest


def evaluate(case: dict, envelope: dict) -> dict:
    try:
        if envelope["httpStatus"] != 200:
            raise ValueError("non-200 model response")
        body = json.loads(envelope["body"])
        raw = json.loads(body["message"]["content"])
        tree = FlowTree.model_validate(raw)
        compile_report(sources(), tree)
        return {"parseAndCompile": True, "exactCopy": sha256_json(raw) == sha256_json(case["expected"]),
            "doneReason": body.get("done_reason"), "inputTokens": body.get("prompt_eval_count"),
            "outputTokens": body.get("eval_count"), "latencyMs": envelope["latencyMs"]}
    except (ValueError, KeyError, TypeError) as error:
        return {"parseAndCompile": False, "exactCopy": False, "errorType": type(error).__name__, "error": str(error)}


def run(root: Path) -> None:
    manifest = load(root)
    for case in manifest["cases"]:
        folder = root / case["id"]
        if folder.exists():
            verify_receipt(folder)
            continue
        model = OllamaAnchoredAuthorAdapter().preflight()
        if model != manifest["model"]:
            raise ValueError("model identity drift")
        folder.mkdir()
        _write(folder / "request.json", {"wireRequest": case["wireRequest"], "model": model})
        start = time.monotonic()
        try:
            with httpx.Client(timeout=180, trust_env=False) as client:
                response = client.post("http://127.0.0.1:11434/api/chat", json=case["wireRequest"])
            envelope = {"httpStatus": response.status_code, "body": response.text, "latencyMs": (time.monotonic() - start) * 1000}
            _write(folder / "response.json", envelope)
            result = evaluate(case, envelope)
        except Exception as error:
            result = {"transportError": type(error).__name__, "error": str(error)}
        _write(folder / "result.json", result)
        _write(folder / "receipt.json", receipt(folder))
        print(json.dumps({"id": case["id"], **result}), flush=True)


def report(root: Path) -> dict:
    manifest = load(root)
    rows = []
    for case in manifest["cases"]:
        folder = root / case["id"]
        verify_receipt(folder)
        if json.loads((folder / "request.json").read_text()) != {"wireRequest": case["wireRequest"], "model": manifest["model"]}:
            raise ValueError("actual request/model drift")
        result = evaluate(case, json.loads((folder / "response.json").read_text()))
        if result != json.loads((folder / "result.json").read_text()):
            raise ValueError("canary result replay mismatch")
        rows.append({"id": case["id"], "arm": case["arm"], "probe": case["probe"], **result, "files": receipt(folder)})
    body = {"protocol": PROTOCOL, "manifestDigest": manifest["manifestDigest"], "model": manifest["model"],
        "rows": rows, "exactCopiesByArm": {arm: sum(row["exactCopy"] for row in rows if row["arm"] == arm) for arm in sorted({row["arm"] for row in rows})},
        "evidenceRole": manifest["evidenceRole"], "uniqueSkills": 0, "runtimeExecutions": 0,
        "boundary": "Explicit answers are model-visible in copying probes. Not source fidelity, Skill accuracy, independent review or Runtime safety. Single attempts in fixed order do not establish a causal performance estimate."}
    return {**body, "reportDigest": sha256_json(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "report"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze(args.root)["manifestDigest"])
    elif args.command == "run":
        run(args.root)
    elif args.output:
        _write(args.output, report(args.root))
    else:
        parser.error("report requires --output")


if __name__ == "__main__":
    main()
