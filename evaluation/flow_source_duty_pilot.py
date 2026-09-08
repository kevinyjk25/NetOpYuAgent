"""Single-attempt 9B source-only probe; no business tools or source-script execution."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import httpx
from jsonschema import ValidationError

from evaluation.flow_node_evidence_pilot import implementation as previous_implementation
from evaluation.flow_source_duties import (
    PROTOCOL, SourceBundle, SourceDuties, assess_duties, compile_duties, request, seal,
)
from evaluation.flow_translation import _write
from evaluation.flow_tree_authoring import ROOT, digest_file, receipt, verify_receipt
from evaluation.flow_tree_bounded_pilot import environment
from evaluation.read_l05_review import ReadL05Review
from evaluation.translation_case_authoring import OllamaAnchoredAuthorAdapter
from network_runtime.contracts import sha256_json


def implementation():
    paths = set(previous_implementation()) | {
        "evaluation/flow_source_duties.py", "evaluation/flow_duty_binding.py", "evaluation/flow_source_duty_pilot.py"}
    return {path: digest_file(ROOT / path) for path in sorted(paths)}


def cases_at(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 32:
        raise ValueError("require 1..32 explicit development cases")
    cases, seen = [], set()
    for item in value:
        key = item["id"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", key) or key in seen:
            raise ValueError("invalid/duplicate case ID")
        seen.add(key)
        bundle = SourceBundle.model_validate(item["bundle"])
        cases.append(dict(id=key, bundle=bundle.model_dump(mode="json"), request=request(bundle)))
    return cases


def freeze(inputs: Path, root: Path):
    cases = cases_at(json.loads(inputs.read_text()))
    manifest = seal(dict(protocol=PROTOCOL, cases=cases,
        model=OllamaAnchoredAuthorAdapter().preflight(), implementation=implementation(), environment=environment(),
        evidenceRole="known_development_source_only_not_holdout_or_whole_translation", attemptsPerCase=1,
        inputFileDigest=digest_file(inputs), publicSkills=0, runtimeExecutions=0), "manifestDigest")
    root.mkdir(parents=True, exist_ok=False)
    _write(root / "manifest.json", manifest)
    return manifest


def load(root):
    m = json.loads((root / "manifest.json").read_text())
    if (m["manifestDigest"] != sha256_json({k: v for k, v in m.items() if k != "manifestDigest"}) or
            m["protocol"] != PROTOCOL or m["implementation"] != implementation() or m["environment"] != environment()):
        raise ValueError("frozen manifest, implementation or environment drift")
    if cases_at(m["cases"]) != m["cases"]:
        raise ValueError("frozen request drift")
    return m


def derive(bundle, envelope):
    files, cost = {}, dict(latencyMs=envelope["latencyMs"])
    if type(cost["latencyMs"]) not in (float, int) or not math.isfinite(cost["latencyMs"]) or cost["latencyMs"] < 0:
        raise ValueError("invalid recorded latency")
    try:
        if envelope["httpStatus"] != 200:
            raise ValueError("transport/non-200 response; original envelope retained")
        reply = json.loads(envelope["body"])
        cost["doneReason"] = reply.get("done_reason")
        for field, token_key in (("inputTokens", "prompt_eval_count"), ("outputTokens", "eval_count")):
            value = reply.get(token_key)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("invalid token accounting")
            cost[field] = value
        if reply.get("model") != "qwen3.5:9b" or reply.get("done") is not True or reply.get("done_reason") != "stop":
            raise ValueError("unexpected model or incomplete generation")
        raw = json.loads(reply["message"]["content"])
        files["raw-candidate.json"] = raw
        candidate = SourceDuties.model_validate(raw)
        compiled = compile_duties(bundle, candidate)
        files.update({"candidate.json": candidate.model_dump(mode="json"),
            "compilation.json": compiled, "review-input.json": compiled["reviewInput"]})
        status = dict(status="awaiting_source_review", inputDigest=compiled["reviewInput"]["inputDigest"])
    except (ValueError, KeyError, TypeError, ValidationError) as error:
        status = dict(status="blocked", errorType=type(error).__name__, error=str(error))
    return files, {**status, **cost}


def replay(folder, case, model):
    verify_receipt(folder)
    if json.loads((folder / "request.json").read_text()) != dict(wireRequest=case["request"], model=model):
        raise ValueError("actual request/model drift")
    files, status = derive(SourceBundle.model_validate(case["bundle"]), json.loads((folder / "response.json").read_text()))
    if (set(receipt(folder)) != {"request.json", "response.json", "status.json", *files} or
            json.loads((folder / "status.json").read_text()) != status or
            any(json.loads((folder / name).read_text()) != value for name, value in files.items())):
        raise ValueError("saved derivation differs from original response")
    return files, status


def run(root, max_new_calls):
    m = load(root)
    pending = [c for c in m["cases"] if not (root / c["id"]).exists()]
    if isinstance(max_new_calls, bool) or not isinstance(max_new_calls, int) or max_new_calls < len(pending):
        raise ValueError("explicit new-call budget insufficient")
    # Verify all old checkpoints before any new request. Uncertain folders never retry.
    for case in m["cases"]:
        if (root / case["id"]).exists():
            replay(root / case["id"], case, m["model"])
    for case in pending:
        if OllamaAnchoredAuthorAdapter().preflight() != m["model"]:
            raise ValueError("model artifact drift")
        folder = root / case["id"]
        folder.mkdir()
        _write(folder / "request.json", dict(wireRequest=case["request"], model=m["model"]))
        start = time.monotonic()
        try:
            with httpx.Client(timeout=360, trust_env=False) as client:
                response = client.post("http://127.0.0.1:11434/api/chat", json=case["request"])
            envelope = dict(httpStatus=response.status_code, body=response.text, latencyMs=(time.monotonic() - start) * 1000)
        except httpx.HTTPError as error:
            envelope = dict(httpStatus=None, body=f"{type(error).__name__}: {error}", latencyMs=(time.monotonic() - start) * 1000)
        _write(folder / "response.json", envelope)
        files, status = derive(SourceBundle.model_validate(case["bundle"]), envelope)
        for name, value in files.items():
            _write(folder / name, value)
        _write(folder / "status.json", status)
        _write(folder / "receipt.json", receipt(folder))
        print(json.dumps(dict(case=case["id"], **status)), flush=True)
        if envelope["httpStatus"] != 200 or status.get("doneReason") != "stop":
            raise ValueError("transport/incomplete response; stop batch without retry")


def report(root, reviews=None):
    m, rows = load(root), []
    for case in m["cases"]:
        folder = root / case["id"]
        if not folder.exists():
            rows.append(dict(case=case["id"], status="not_run"))
            continue
        files, status = replay(folder, case, m["model"])
        row = dict(case=case["id"], **status, files=receipt(folder))
        if reviews and status["status"] == "awaiting_source_review":
            review = ReadL05Review.model_validate_json((reviews / (case["id"] + ".json")).read_text())
            row["review"] = assess_duties(SourceBundle.model_validate(case["bundle"]),
                SourceDuties.model_validate(files["candidate.json"]), review)
        rows.append(row)
    latencies = sorted(row["latencyMs"] for row in rows if "latencyMs" in row)
    return seal(dict(manifestDigest=m["manifestDigest"], model=m["model"], evidenceRole=m["evidenceRole"], rows=rows,
        completed=all(r["status"] != "not_run" for r in rows), knownDevelopmentCases=len(rows), publicSkills=0,
        structurallyQualified=sum(r["status"] == "awaiting_source_review" for r in rows),
        sourceReviewed=sum("review" in r for r in rows),
        reviewSupported=sum(r.get("review", {}).get("status") == "review_supported_source_candidates" for r in rows),
        postTotalMs=sum(latencies), p50Ms=latencies[math.ceil(len(latencies) * .5) - 1] if latencies else None,
        p95Ms=latencies[math.ceil(len(latencies) * .95) - 1] if latencies else None,
        inputTokens=sum(r.get("inputTokens") or 0 for r in rows), outputTokens=sum(r.get("outputTokens") or 0 for r in rows),
        semanticAccuracy=None, runtimeExecutions=0, wholeSkillTranslations=0, runtimeAuthorityGranted=False), "reportDigest")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["freeze", "run", "report"])
    p.add_argument("root", type=Path)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--reviews", type=Path)
    p.add_argument("--max-new-calls", type=int, default=0)
    a = p.parse_args()
    if a.command == "freeze":
        if not a.inputs:
            p.error("freeze requires --inputs")
        print(freeze(a.inputs, a.root)["manifestDigest"])
    elif a.command == "run":
        run(a.root, a.max_new_calls)
    else:
        if not a.output:
            p.error("report requires --output")
        _write(a.output, report(a.root, a.reviews))


if __name__ == "__main__":
    main()
