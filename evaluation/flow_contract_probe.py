"""Compare contract constructors with an unchanged, frozen behavior probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import ValidationError

from evaluation import flow_behavior_probe as parent
from evaluation.flow_behavior import _seal
from evaluation.flow_checkpoint import replay as replay_checkpoint
from evaluation.flow_contract_authoring import PROTOCOL, constructor_request, lower_constructors
from evaluation.flow_translation import FlowSources, _write
from evaluation.flow_tree_authoring import digest_file, receipt


def inputs_for(manifest):
    return dict(protocol=PROTOCOL, parentManifestDigest=manifest["reportDigest"], model=manifest["model"],
        implementation={name: digest_file(Path(__file__).with_name(name))
            for name in ("flow_contract_authoring.py", "flow_contract_probe.py")},
        requests={c["id"]: constructor_request(FlowSources.model_validate(c["sources"])) for c in manifest["cases"]})


def derive(case, envelope):
    text, status = parent.decode("ollama", envelope)
    files = {}
    if text is not None:
        try:
            raw = json.loads(text)
            files["proposal.json"] = raw
            lowered = lower_constructors(FlowSources.model_validate(case["sources"]), raw)
            files["lowering.json"] = lowered
            files["candidate.json"] = lowered["tree"]
            files["behavior.json"] = parent.evaluate(case, lowered["tree"])
            status["candidateStatus"] = files["behavior.json"]["behavior"]
        except (ValueError, KeyError, TypeError, ValidationError) as error:
            status.update(candidateStatus="invalid_candidate", errorType=type(error).__name__, error=str(error)[:3000])
    return files, status


def replay(folder, case, inputs):
    return replay_checkpoint(folder, dict(wireRequest=inputs["requests"][case["id"]], model=inputs["model"]),
        lambda envelope: derive(case, envelope), label="constructor")


def load(root, parent_root):
    manifest = parent.load(parent_root)
    inputs = inputs_for(manifest)
    if root.exists() and json.loads((root / "inputs.json").read_text()) != inputs:
        raise ValueError("frozen constructor implementation/request drift")
    return manifest, inputs


def run(root, parent_root, max_new_calls):
    manifest, inputs = load(root, parent_root)
    pending = [c for c in manifest["cases"] if not (root / c["id"]).exists()]
    if type(max_new_calls) is not int or max_new_calls < len(pending):
        raise ValueError("explicit call budget insufficient")
    for case in manifest["cases"]:
        if (root / case["id"]).exists():
            _, result = replay(root / case["id"], case, inputs)
            if result["status"] != "text_received" and pending:
                raise ValueError("prior transport failure; no retry or automatic continuation")
    if not root.exists():
        root.mkdir(parents=True)
        _write(root / "inputs.json", inputs)
    for case in pending:
        if parent.OllamaAnchoredAuthorAdapter().preflight() != manifest["model"]:
            raise ValueError("model artifact drift")
        folder = root / case["id"]
        folder.mkdir()
        _write(folder / "request.json", dict(wireRequest=inputs["requests"][case["id"]], model=inputs["model"]))
        envelope = parent.send("ollama", inputs["requests"][case["id"]])
        _write(folder / "response.json", envelope)
        files, result = derive(case, envelope)
        for name, value in files.items():
            _write(folder / name, value)
        _write(folder / "result.json", result)
        _write(folder / "receipt.json", receipt(folder))
        print(json.dumps(dict(id=case["id"], **result)), flush=True)
        if result["status"] != "text_received":
            raise ValueError("stop after transport/envelope failure; never retry")


def report(root, parent_root):
    manifest, inputs = load(root, parent_root)
    rows = []
    for case in manifest["cases"]:
        row = dict(id=case["id"], scope=case["suite"]["scope"], status="not_run", total=len(case["suite"]["scenarios"]))
        if (root / case["id"]).exists():
            files, result = replay(root / case["id"], case, inputs)
            row.update(result, behavior=files.get("behavior.json"), files=receipt(root / case["id"]))
        rows.append(row)
    return _seal(dict(**inputs, rows=rows, completed=all(r["status"] != "not_run" for r in rows),
        matched=sum((r.get("behavior") or {}).get("behavior") == "matched_finite_oracle" for r in rows),
        passedScenarios=sum((r.get("behavior") or {}).get("passed", 0) for r in rows),
        totalScenarios=sum(r["total"] for r in rows),
        inputTokens=sum(r.get("inputTokens") or 0 for r in rows), outputTokens=sum(r.get("outputTokens") or 0 for r in rows),
        postTotalMs=sum(r.get("latencyMs") or 0 for r in rows),
        wholeSkillTranslations=0, semanticAccuracy=None, publicSkills=0, runtimeAuthorityGranted=False,
        boundary="Same known development cases and private finite oracles; no whole-Skill semantic acceptance."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--max-new-calls", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.root, args.parent, args.max_new_calls)
    elif args.output:
        _write(args.output, report(args.root, args.parent))
    else:
        parser.error("report requires --output")


if __name__ == "__main__":
    main()
