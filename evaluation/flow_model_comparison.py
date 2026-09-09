"""Frozen GPT/9B diagnostic: fixed-tree mapping or fresh end-to-end proposals only."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from evaluation import flow_model_transport as transport
from evaluation import flow_node_evidence_pilot as historical
from evaluation.flow_node_evidence import EvidenceMapping, assess_evidence, compile_evidence, projection, request
from evaluation.flow_translation import FlowSources, _write
from evaluation.flow_tree import FlowTree
from evaluation.flow_tree_authoring import ROOT, digest_file, receipt, verify_receipt
from evaluation.flow_tree_bounded_pilot import environment
from evaluation.flow_tree_capabilities import bounded_request, validate_bounded
from evaluation.read_l05_review import ReadL05Review
from evaluation.translation_case_authoring import OllamaAnchoredAuthorAdapter
from network_runtime.contracts import sha256_json

PROTOCOL = "flow-model-common-json/v1"
ARMS = ("openai", "ollama")


def implementation():
    return {**historical.implementation(), **{p: digest_file(ROOT / p) for p in (
        "evaluation/flow_model_transport.py", "evaluation/flow_model_comparison.py")}}


def sealed(body, field):
    return {**body, field: sha256_json(body)}


def verify_seal(value, field):
    if value[field] != sha256_json({k: v for k, v in value.items() if k != field}):
        raise ValueError(field + " mismatch")


def read_json(path):
    return json.loads(path.read_text())


def freeze(source_manifest: Path, root: Path, mode="end-to-end", repeats=1, parent_run=None):
    if mode not in {"end-to-end", "mapping-only"} or type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError("invalid mode/repeats")
    if (parent_run is not None) != (mode == "mapping-only"):
        raise ValueError("parent run required only for mapping-only; never supply old trees to end-to-end")
    src = read_json(source_manifest)
    verify_seal(src, "manifestDigest")
    parent_manifest = historical.load(parent_run) if parent_run else None
    parent_cases = {c["id"]: c for c in parent_manifest["cases"]} if parent_manifest else {}
    rows = []
    for case in src["cases"]:
        key = case["id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", key) or key in {r["id"] for r in rows}:
            raise ValueError("invalid/duplicate case ID")
        source = FlowSources.model_validate(case["sources"])
        row = dict(id=key, sources=source.model_dump(mode="json"))
        if parent_run:
            p = parent_cases[key]
            if p["sources"] != row["sources"]:
                raise ValueError("parent source mismatch")
            folder = parent_run / key / "flow"
            files, status = historical.replay(folder, source, None, p["flowRequest"], parent_manifest["model"])
            if status["status"] != "awaiting_source_review":
                raise ValueError("mapping-only requires qualified fixed tree")
            row.update(fixedTree=files["tree.json"], parentFlowReceipt=receipt(folder))
        rows.append(row)
    if not rows or len(rows) > 12:
        raise ValueError("diagnostic requires 1–12 development cases")
    body = dict(protocol=PROTOCOL, mode=mode, repeats=repeats, cases=rows,
        sourceManifestDigest=src["manifestDigest"],
        parentManifestDigest=parent_manifest["manifestDigest"] if parent_manifest else None,
        implementation=implementation(), environment=environment(),
        arms=dict(openai=dict(model=transport.GPT_MODEL, reasoning="high"),
                  ollama=dict(model=transport.QWEN_MODEL, think=False)),
        maxOutputTokens=transport.MAX_OUTPUT, attemptsPerPhase=1,
        evidenceRole="known_development_model_provider_configuration_diagnostic_not_holdout",
        decoding="same full schema in prompt; JSON mode on both; unchanged local validator",
        reviewPolicy="separate digest-bound source review; no automatic Gold or accuracy",
        limitations=["provider/reasoning/decoder/tokenizer differences remain; not weights-only causality",
                     "old constrained-decoder 9B scores are historical, not the new control arm",
                     "fixed-tree mapping is conditional on an unreviewed historical candidate",
                     "no business execution, activation, scripts or tools"])
    manifest = sealed(body, "manifestDigest")
    root.mkdir(parents=True, exist_ok=False)
    _write(root / "manifest.json", manifest)
    return manifest


def load(root):
    m = read_json(root / "manifest.json")
    verify_seal(m, "manifestDigest")
    if m["protocol"] != PROTOCOL or m["implementation"] != implementation() or m["environment"] != environment():
        raise ValueError("frozen implementation/environment drift")
    return m


def derive(source, parent, arm, original, envelope):
    text, status = transport.decode(arm, envelope)
    files = {}
    if text is None:
        return files, status
    try:
        raw = json.loads(text)
        json.dumps(raw, allow_nan=False)
        files["proposal.json"] = raw
        Draft202012Validator(original["format"]).validate(raw)
        status["schemaQualified"] = True
        if parent is None:
            tree = FlowTree.model_validate(raw)
            compiled = validate_bounded(source, raw)
            files["tree.json"] = tree.model_dump(mode="json")
        else:
            proposal = EvidenceMapping.model_validate(raw)
            files["diagnostics.json"] = projection(source, parent, proposal)[1]
            compiled = compile_evidence(source, parent, proposal)
            files["mapping.json"] = proposal.model_dump(mode="json")
        files["compilation.json"] = compiled
        files["review-input.json"] = compiled["reviewInput"]
        status.update(status="awaiting_source_review", inputDigest=compiled["reviewInput"]["inputDigest"])
    except (ValueError, KeyError, TypeError, ValidationError) as error:
        status.update(status="blocked", errorType=type(error).__name__, error=str(error))
    return files, status


def save_once(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError("checkpoint derivation mismatch: " + str(path))
    else:
        _write(path, value)


def materialize(folder, source, parent, arm, original, expected, recover=False):
    if read_json(folder / "request.json") != expected:
        raise ValueError("request binding drift")
    # A missing response may have been charged remotely. Never silently retry it.
    if not (folder / "response.json").exists():
        return {}, {"status": "execution_uncertain", "latencyMs": None}
    files, status = derive(source, parent, arm, original, read_json(folder / "response.json"))
    expected_files = {"request.json", "response.json", "status.json", *files}
    if recover and not (folder / "receipt.json").exists():
        if set(receipt(folder)) - expected_files:
            raise ValueError("unexpected checkpoint files")
        for name, value in {**files, "status.json": status}.items():
            save_once(folder / name, value)
        _write(folder / "receipt.json", receipt(folder))
    verify_receipt(folder)
    if (set(receipt(folder)) != expected_files or read_json(folder / "status.json") != status or
            any(read_json(folder / name) != value for name, value in files.items())):
        raise ValueError("raw response derivation mismatch")
    return files, status


def ensure_continuable(status):
    if status["status"] in {"execution_uncertain", "model_identity_mismatch", "transport_error"}:
        raise ValueError("model run stopped: " + status["status"] + "; inspect checkpoint, do not retry the attempt")


def phase(root, m, case, repeat, arm, name, parent, execute=False, budget=None):
    source = FlowSources.model_validate(case["sources"])
    original = bounded_request(source) if parent is None else request(source, parent)
    wire = transport.wire_request(original, arm, repeat)
    folder = root / arm / str(repeat) / case["id"] / name
    identity_path = root / "ollama-identity.json"
    if not folder.exists() and execute and budget[0] and arm == "ollama":
        save_once(identity_path, OllamaAnchoredAuthorAdapter().preflight())
    identity = read_json(identity_path) if arm == "ollama" and identity_path.exists() else None
    expected = dict(manifestDigest=m["manifestDigest"], case=case["id"], repeat=repeat, arm=arm,
                    phase=name, originalRequestDigest=sha256_json(original), wireRequest=wire,
                    localModelIdentity=identity)
    if folder.exists():
        return materialize(folder, source, parent, arm, original, expected, recover=execute)
    if not execute or budget[0] == 0:
        return {}, {"status": "not_run", "latencyMs": None}
    transport.check_credentials(arm)
    folder.mkdir(parents=True, exist_ok=False)
    _write(folder / "request.json", expected)
    budget[0] -= 1
    _write(folder / "response.json", transport.send(arm, wire))
    return materialize(folder, source, parent, arm, original, expected, recover=True)


def walk(root, m, arm, execute=False, budget=None, reviews=None):
    rows = []
    for repeat in range(m["repeats"]):
        for case in m["cases"]:
            row = dict(case=case["id"], repeat=repeat)
            parent = FlowTree.model_validate(case["fixedTree"]) if m["mode"] == "mapping-only" else None
            if parent is None:
                files, status = phase(root, m, case, repeat, arm, "flow", None, execute, budget)
                row["flow"] = status
                if status["status"] != "awaiting_source_review":
                    rows.append(row)
                    if execute:
                        ensure_continuable(status)
                    continue
                parent = FlowTree.model_validate(files["tree.json"])
            files, status = phase(root, m, case, repeat, arm, "mapping", parent, execute, budget)
            row["mapping"] = status
            if execute:
                ensure_continuable(status)
            if reviews and status["status"] == "awaiting_source_review":
                path = reviews / arm / str(repeat) / (case["id"] + ".json")
                if path.exists():
                    review = read_json(path)
                    context = dict(manifestDigest=m["manifestDigest"], arm=arm, repeat=repeat, case=case["id"])
                    if review.get("context") != context:
                        raise ValueError("review experiment/arm/case binding mismatch")
                    row["review"] = assess_evidence(FlowSources.model_validate(case["sources"]), parent,
                        EvidenceMapping.model_validate(files["mapping.json"]), ReadL05Review.model_validate(review["review"]))
            rows.append(row)
    return rows


def run(root, arm, max_calls):
    if arm not in ARMS or type(max_calls) is not int or not 1 <= max_calls <= 24:
        raise ValueError("explicit arm and max-calls (1–24) required")
    return walk(root, load(root), arm, execute=True, budget=[max_calls])


def percentile(values, q):
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)] if values else None


def report(root, reviews=None):
    m = load(root)
    arms = {}
    for arm in ARMS:
        rows = walk(root, m, arm, reviews=reviews)
        phases = [row[name] for row in rows for name in ("flow", "mapping") if name in row]
        received = [p for p in phases if p["status"] not in {"not_run", "execution_uncertain"}]
        latencies = [p["latencyMs"] for p in received]
        complete = all(row.get("mapping", row.get("flow", {})).get("status") not in
                       {"not_run", "execution_uncertain"} for row in rows)
        arms[arm] = dict(rows=rows, completed=complete, recordedResponses=len(received),
            pendingOrUncertain=sum(p["status"] in {"not_run", "execution_uncertain"} for p in phases),
            flowQualified=sum(row.get("flow", {}).get("status") == "awaiting_source_review" for row in rows)
                          if m["mode"] == "end-to-end" else None,
            mappingQualified=sum(row.get("mapping", {}).get("status") == "awaiting_source_review" for row in rows),
            sourceReviewed=sum("review" in row for row in rows),
            reviewSupportedInactiveFlows=sum(row.get("review", {}).get("status") == "review_supported_inactive_flow" for row in rows),
            semanticAccuracy=None, generalizationAccuracy=None, unsafeAcceptRate=None,
            latencyMs=sum(latencies), requestP50Ms=percentile(latencies, .5), requestP95Ms=percentile(latencies, .95),
            inputTokens=sum(p.get("inputTokens") or 0 for p in received),
            outputTokens=sum(p.get("outputTokens") or 0 for p in received),
            tokenCountsComplete=bool(received) and all(p.get("inputTokens") is not None and p.get("outputTokens") is not None for p in received))
    body = dict(protocol=PROTOCOL, manifestDigest=m["manifestDigest"], mode=m["mode"],
        knownFlows=len(m["cases"]), publicSkills=0, repeats=m["repeats"], arms=arms,
        modelConfigurations=m["arms"],
        localModelIdentity=read_json(root / "ollama-identity.json") if (root / "ollama-identity.json").exists() else None,
        status="mechanical_results_need_separate_review" if all(a["completed"] for a in arms.values()) else "pending_model_calls",
        decision="not_established", runtimeExecutions=0, businessWrites=0,
        boundary="No model winner inferred from compilation. GPT review is not Gold. Partial/not-run is not zero accuracy; latency is per POST, not Runtime performance.")
    return sealed(body, "reportDigest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "report"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("end-to-end", "mapping-only"), default="end-to-end")
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    if args.command != "run" and args.output is None:
        parser.error("freeze/report requires --output")
    if args.command == "freeze":
        result = freeze(args.root, args.output, args.mode, args.repeats, args.parent_run)
        print(result["manifestDigest"])
    elif args.command == "run":
        for row in run(args.root, args.arm, args.max_calls):
            print(json.dumps(row, ensure_ascii=False))
    else:
        _write(args.output, report(args.root, args.reviews))


if __name__ == "__main__":
    main()
