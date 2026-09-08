"""Frozen guarded-source probe; explicit cohorts, original responses, zero execution."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import httpx
from jsonschema import ValidationError

from evaluation import flow_source_duty_pilot as previous
from evaluation import flow_source_guards as protocol
from evaluation.flow_source_duties import SourceBundle, seal
from evaluation.flow_translation import _write
from evaluation.flow_tree_authoring import ROOT, digest_file, receipt, verify_receipt
from evaluation.read_l05_review import ReadL05Review
from evaluation.translation_case_authoring import OllamaAnchoredAuthorAdapter
from network_runtime.contracts import sha256_json


def implementation():
    paths = set(previous.implementation()) | {
        "evaluation/flow_source_guards.py", "evaluation/flow_source_guard_pilot.py"}
    return {p: digest_file(ROOT / p) for p in sorted(paths)}


def cases_at(value):
    rows = previous.cases_at(value)  # Reuse source/identifier bounds, not its v1 request.
    for row, item in zip(rows, value):
        if item.get("cohort") not in {"regression", "new_development"}:
            raise ValueError("explicit development cohort required")
        row.update(cohort=item["cohort"], request=protocol.request(SourceBundle.model_validate(row["bundle"])))
    return rows


def freeze(inputs, root):
    cases = cases_at(json.loads(inputs.read_text()))
    body = dict(protocol=protocol.PROTOCOL, cases=cases, implementation=implementation(),
        environment=previous.environment(), model=OllamaAnchoredAuthorAdapter().preflight(),
        inputFileDigest=digest_file(inputs), attemptsPerCase=1, publicSkills=0,
        evidenceRole="known_development_guarded_source_only_not_gold_or_whole_translation")
    manifest = seal(body, "manifestDigest")
    root.mkdir(parents=True, exist_ok=False)
    _write(root / "manifest.json", manifest)
    return manifest


def load(root):
    m = json.loads((root / "manifest.json").read_text())
    if (m["manifestDigest"] != sha256_json({k: v for k, v in m.items() if k != "manifestDigest"}) or
            m["protocol"] != protocol.PROTOCOL or m["implementation"] != implementation() or
            m["environment"] != previous.environment() or m["cases"] != cases_at(m["cases"])):
        raise ValueError("frozen source/request/implementation/environment drift")
    return m


def derive(bundle, envelope):
    files = {}
    latency = envelope["latencyMs"]
    if type(latency) not in (float, int) or not math.isfinite(latency) or latency < 0:
        raise ValueError("invalid latency")
    cost = dict(latencyMs=latency, envelopeValid=False)
    try:
        if envelope["httpStatus"] != 200:
            raise ValueError("transport failure or non-200 response")
        reply = json.loads(envelope["body"])
        if not isinstance(reply, dict):
            raise ValueError("model envelope must be an object")
        cost["doneReason"] = reply.get("done_reason")
        for key, field in (("inputTokens", "prompt_eval_count"), ("outputTokens", "eval_count")):
            value = reply.get(field)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("invalid token accounting")
            cost[key] = value
        if reply.get("model") != "qwen3.5:9b" or reply.get("done") is not True or reply.get("done_reason") != "stop":
            raise ValueError("model identity mismatch or incomplete generation")
        if not isinstance(reply.get("message"), dict) or not isinstance(reply["message"].get("content"), str):
            raise ValueError("invalid assistant message")
        cost["envelopeValid"] = True
        raw = json.loads(reply["message"]["content"])
        files["raw-candidate.json"] = raw
        candidate = protocol.GuardedDuties.model_validate(raw)
        compiled = protocol.compile_candidate(bundle, candidate)
        files.update({"candidate.json": candidate.model_dump(mode="json"), "compilation.json": compiled,
            "review-input.json": compiled["reviewInput"]})
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
        raise ValueError("raw derivation drift")
    return files, status


def run(root, max_new_calls):
    m = load(root)
    pending = [c for c in m["cases"] if not (root / c["id"]).exists()]
    if type(max_new_calls) is not int or max_new_calls < len(pending):
        raise ValueError("explicit new-call budget insufficient")
    for case in m["cases"]:
        if (root / case["id"]).exists():
            _, status = replay(root / case["id"], case, m["model"])
            if not status["envelopeValid"] and pending:
                raise ValueError("prior transport/envelope failure; no automatic continuation")
    for case in pending:
        if OllamaAnchoredAuthorAdapter().preflight() != m["model"]:
            raise ValueError("model artifact drift")
        folder = root / case["id"]
        folder.mkdir()
        _write(folder / "request.json", dict(wireRequest=case["request"], model=m["model"]))
        started = time.monotonic()
        try:
            with httpx.Client(timeout=360, trust_env=False) as client:
                r = client.post("http://127.0.0.1:11434/api/chat", json=case["request"])
            envelope = dict(httpStatus=r.status_code, body=r.text, latencyMs=(time.monotonic() - started) * 1000)
        except httpx.HTTPError as error:
            envelope = dict(httpStatus=None, body=str(error), latencyMs=(time.monotonic() - started) * 1000)
        _write(folder / "response.json", envelope)
        files, status = derive(SourceBundle.model_validate(case["bundle"]), envelope)
        for name, value in files.items():
            _write(folder / name, value)
        _write(folder / "status.json", status)
        _write(folder / "receipt.json", receipt(folder))
        print(json.dumps(dict(case=case["id"], **status)), flush=True)
        if not status["envelopeValid"]:
            raise ValueError("stop batch on transport/envelope failure; never retry")


def report(root, reviews=None):
    m, rows = load(root), []
    for case in m["cases"]:
        folder = root / case["id"]
        row = dict(case=case["id"], cohort=case["cohort"], status="not_run")
        if folder.exists():
            files, status = replay(folder, case, m["model"])
            row.update(**status, files=receipt(folder))
            if reviews and status["status"] == "awaiting_source_review":
                review = ReadL05Review.model_validate_json((reviews / (case["id"] + ".json")).read_text())
                row["review"] = protocol.assess(SourceBundle.model_validate(case["bundle"]),
                    protocol.GuardedDuties.model_validate(files["candidate.json"]), review)
        rows.append(row)
    cohorts = {}
    for name in sorted({r["cohort"] for r in rows}):
        group = [r for r in rows if r["cohort"] == name]
        times = sorted(r["latencyMs"] for r in group if "latencyMs" in r)
        cohorts[name] = dict(cases=len(group), structurallyQualified=sum(r["status"] == "awaiting_source_review" for r in group),
            sourceReviewed=sum("review" in r for r in group),
            reviewSupported=sum(r.get("review", {}).get("status") == "review_supported_source_candidates" for r in group),
            postTotalMs=sum(times), p50Ms=times[math.ceil(len(times) * .5) - 1] if times else None,
            p95Ms=times[math.ceil(len(times) * .95) - 1] if times else None,
            inputTokens=sum(r.get("inputTokens") or 0 for r in group), outputTokens=sum(r.get("outputTokens") or 0 for r in group),
            tokenAccountingComplete=all(type(r.get("inputTokens")) is int and type(r.get("outputTokens")) is int for r in group))
    return seal(dict(manifestDigest=m["manifestDigest"], model=m["model"], evidenceRole=m["evidenceRole"], rows=rows,
        completed=all(r["status"] != "not_run" for r in rows), cohorts=cohorts, semanticAccuracy=None,
        publicSkills=0, wholeSkillTranslations=0, runtimeExecutions=0, runtimeAuthorityGranted=False), "reportDigest")


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
