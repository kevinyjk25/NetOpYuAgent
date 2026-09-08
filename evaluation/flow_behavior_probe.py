"""One-shot 9B flow generation with hidden finite oracles and inert execution.

freeze checks manual feasibility first. run never sends a witness, observations,
expected outcome, case ID or review verdict to the model. No automatic repair.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evaluation import flow_behavior as behavior
from evaluation.flow_behavior_examples import cases
from evaluation.flow_model_transport import decode, send
from evaluation.flow_checkpoint import environment, implementation as shared_implementation
from evaluation.flow_checkpoint import replay as replay_checkpoint
from evaluation.flow_translation import FlowSources, _write
from evaluation.flow_tree import FlowTree
from evaluation.flow_tree_authoring import receipt
from evaluation.translation_case_authoring import OllamaAnchoredAuthorAdapter


def implementation():
    return shared_implementation("evaluation/flow_behavior_examples.py", "evaluation/flow_behavior_probe.py")


def evaluate(case, raw):
    return behavior.check_behavior(FlowSources.model_validate(case["sources"]), FlowTree.model_validate(raw),
        behavior.BehaviorSuite.model_validate(case["suite"]))


def freeze(root):
    rows = cases()
    for case in rows:
        case["referenceCheck"] = evaluate(case, case["reference"])
        if case["referenceCheck"]["behavior"] != "matched_finite_oracle":
            raise ValueError("manual feasibility witness fails; do not blame or call the model")
        case["request"] = behavior.behavior_request(FlowSources.model_validate(case["sources"]))
    body = dict(protocol=behavior.PROTOCOL, cases=rows, implementation=implementation(), environment=environment(),
        model=OllamaAnchoredAuthorAdapter().preflight(), attemptsPerCase=1,
        evidenceRole="known_development_finite_behavior_not_independent_gold", publicSkills=0)
    manifest = behavior._seal(body)
    root.mkdir(parents=True, exist_ok=False)
    _write(root / "manifest.json", manifest)
    return manifest


def load(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest != behavior._seal({k: v for k, v in manifest.items() if k != "reportDigest"})
            or manifest["implementation"] != implementation() or manifest["environment"] != environment()
            or manifest["protocol"] != behavior.PROTOCOL):
        raise ValueError("frozen behavior protocol/implementation/environment drift")
    expected = cases()
    if len(expected) != len(manifest["cases"]):
        raise ValueError("frozen case count drift")
    for original, case in zip(expected, manifest["cases"], strict=True):
        if original != {k: v for k, v in case.items() if k not in {"referenceCheck", "request"}}:
            raise ValueError("frozen source/reference/oracle drift")
        if case["request"] != behavior.behavior_request(FlowSources.model_validate(case["sources"])):
            raise ValueError("frozen request drift")
        if case["referenceCheck"] != evaluate(case, case["reference"]):
            raise ValueError("manual feasibility witness drift")
    return manifest


def derive(case, envelope):
    latency = envelope["latencyMs"]
    if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
        raise ValueError("invalid latency")
    text, cost = decode("ollama", envelope)
    files = {}
    if text is not None:
        try:
            raw = json.loads(text)
            files["candidate.json"] = raw
            files["behavior.json"] = evaluate(case, raw)
            cost["candidateStatus"] = files["behavior.json"]["behavior"]
        except (ValueError, KeyError, TypeError) as error:
            cost.update(candidateStatus="invalid_candidate", errorType=type(error).__name__, error=str(error))
    return files, cost


def replay(folder, case, model):
    return replay_checkpoint(folder, dict(wireRequest=case["request"], model=model),
        lambda envelope: derive(case, envelope), label="behavior")


def run(root, max_new_calls):
    manifest = load(root)
    pending = [c for c in manifest["cases"] if not (root / c["id"]).exists()]
    if type(max_new_calls) is not int or max_new_calls < len(pending):
        raise ValueError("explicit call budget is insufficient")
    for case in manifest["cases"]:
        folder = root / case["id"]
        if folder.exists():
            _, result = replay(folder, case, manifest["model"])
            if result["status"] != "text_received" and pending:
                raise ValueError("prior transport failure; no automatic continuation")
    for case in pending:
        if OllamaAnchoredAuthorAdapter().preflight() != manifest["model"]:
            raise ValueError("model artifact drift")
        folder = root / case["id"]
        folder.mkdir()
        _write(folder / "request.json", dict(wireRequest=case["request"], model=manifest["model"]))
        envelope = send("ollama", case["request"])
        _write(folder / "response.json", envelope)
        files, result = derive(case, envelope)
        for name, data in files.items():
            _write(folder / name, data)
        _write(folder / "result.json", result)
        _write(folder / "receipt.json", receipt(folder))
        print(json.dumps(dict(id=case["id"], **result)), flush=True)
        if result["status"] != "text_received":
            raise ValueError("stop after transport/envelope failure; never retry")


def report(root):
    manifest, rows = load(root), []
    for case in manifest["cases"]:
        row = dict(id=case["id"], scope=case["suite"]["scope"], status="not_run")
        if (root / case["id"]).exists():
            files, result = replay(root / case["id"], case, manifest["model"])
            row.update(result, behavior=files.get("behavior.json"), files=receipt(root / case["id"]))
        rows.append(row)
    groups = {}
    for scope in sorted({r["scope"] for r in rows}):
        group = [r for r in rows if r["scope"] == scope]
        groups[scope] = dict(cases=len(group),
            matched=sum(r.get("behavior", {}).get("behavior") == "matched_finite_oracle" for r in group if r.get("behavior")),
            passedScenarios=sum((r.get("behavior") or {}).get("passed", 0) for r in group),
            totalScenarios=sum(len(c["suite"]["scenarios"]) for c in manifest["cases"] if c["suite"]["scope"] == scope))
    times = sorted(r["latencyMs"] for r in rows if "latencyMs" in r)
    return behavior._seal(dict(protocol=behavior.PROTOCOL, manifestDigest=manifest["reportDigest"],
        model=manifest["model"], rows=rows, groups=groups, completed=all(r["status"] != "not_run" for r in rows),
        postTotalMs=sum(times), p50Ms=times[math.ceil(len(times) * .5) - 1] if times else None,
        p95Ms=times[math.ceil(len(times) * .95) - 1] if times else None,
        inputTokens=sum(r.get("inputTokens") or 0 for r in rows), outputTokens=sum(r.get("outputTokens") or 0 for r in rows),
        semanticAccuracy=None, wholeSkillTranslations=0, publicSkills=0, runtimeAuthorityGranted=False,
        boundary="Finite behavior checks on known examples; safe stops are not completed business Skills. Full-source review remains required."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "report"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--max-new-calls", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze(args.root)["reportDigest"])
    elif args.command == "run":
        run(args.root, args.max_new_calls)
    elif args.output:
        _write(args.output, report(args.root))
    else:
        parser.error("report requires --output")


if __name__ == "__main__":
    main()
