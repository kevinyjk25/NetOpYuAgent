"""Explicit two-call 9B diagnostics on retained inputs, never historical regrades.

Usage: python -m evaluation.semantic_witness_probe manifest.json NEW_OUTPUT
       [--max-model-calls 2*N]
The manifest maps case labels to existing, receipt-bound model request files.
Zero is preflight only. Existing output is never overwritten or auto-retried.
"""
import argparse
import json
import re
import tarfile
from pathlib import Path

from evaluation import hybrid_semantic_witness as witness
from evaluation import hybrid_authoring as author
from evaluation.flow_checkpoint import author_once
from evaluation.flow_model_transport import decode
from evaluation.flow_tree_authoring import verify_receipt
from evaluation.hybrid_draft_loop import REVIEW_CONFIG, seal, transport_schema
from evaluation.hybrid_draft_review import build_review_input
from evaluation.semantic_closure_transfer import ROOT, fingerprint
from evaluation.semantic_closure_evidence import collect
from evaluation.source_ledger import budget
from evaluation.structured_binding_probe import read_json, write_artifacts
from network_runtime.contracts import sha256_json
from network_runtime.l0.read_contracts import _source_object


def wire_request(system, inputs, schema):
    return {"model": author.MODEL, "think": False, "stream": False, "format": transport_schema(schema),
        "options": {k: v for k, v in REVIEW_CONFIG.items() if k != "think"},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": json.dumps(inputs, ensure_ascii=False, separators=(",", ":"))}]}


def call(folder, wire, binder):
    measured = budget(wire)
    if not measured["accepted"]:
        write_artifacts(folder / "preflight", {"report.json": seal({"status": "context_budget_exceeded",
            "wireDigest": sha256_json(wire), "budget": measured, "newCalls": 0})})
        return {"result": {"status": "context_budget_exceeded"}}

    def derive(envelope):
        content, cost = decode("ollama", envelope)
        if content is None:
            return {}, cost
        files = {}
        try:
            raw = _source_object(content)
            files["raw-opinion.json"] = raw
            files["bound-opinion.json"] = binder(raw)
            return files, {**cost, "status": "located_unverified_opinion"}
        except (ValueError, TypeError) as error:
            files["invalid-opinion.json"] = {"text": content, "reason": str(error)[:1200]}
            return files, {**cost, "status": "invalid_schema_or_binding"}

    return author_once(folder, {"wireRequest": wire, "wireProfile": witness.PROFILE}, derive,
                       max_new_calls=1, label="candidate-blind semantic witness diagnostic")


def run(manifest, output, *, max_model_calls=0, reuse_plans=None):
    cases = read_json(manifest)
    if (not isinstance(cases, dict) or not 1 <= len(cases) <= 4
            or any(not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", key) for key in cases)):
        raise ValueError("one to four explicitly named retained cases required")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("preserve all attempts; new output required")
    payloads, requests = {}, {}
    for key, path in cases.items():
        path = Path(path).resolve()
        if output.is_relative_to(path.parent):
            raise ValueError("new evidence must be outside historical checkpoint")
        verify_receipt(path.parent)
        request = read_json(path)
        payloads[key] = build_review_input(request["governedRequest"]["inputs"])
        requests[key] = {"path": str(path), "requestDigest": sha256_json(request)}
    retained = {}
    if reuse_plans is not None:
        reuse_plans = Path(reuse_plans).resolve()
        if output.is_relative_to(reuse_plans):
            raise ValueError("recovery must not write into previous run")
        for key, payload in payloads.items():
            previous = reuse_plans / key / "plan"
            if not previous.exists():
                continue
            if (reuse_plans / key / "check").exists():
                raise ValueError("prior comparison already attempted; do not repeat completed or failed checks")
            verify_receipt(previous)
            source = witness.source_input(payload)
            wire = wire_request(witness.PLAN_SYSTEM, witness.plan_input(source), witness.plan_schema(source))
            if read_json(previous / "request.json")["wireRequest"] != wire:
                raise ValueError("retained plan input/model/prompt/schema drift; not reusable")
            text, _ = decode("ollama", read_json(previous / "response.json"))
            if text is None:
                raise ValueError("failed prior plan cannot be reused or retried")
            bound = witness.bind_plan(source, _source_object(text))
            if bound != read_json(previous / "bound-opinion.json"):
                raise ValueError("retained plan derivation drift")
            retained[key] = {"path": str(previous), "receipt": read_json(previous / "receipt.json"),
                             "boundPlan": bound, "result": read_json(previous / "result.json")}
    expected_calls = 2 * len(cases) - len(retained)
    if type(max_model_calls) is not int or max_model_calls not in (0, expected_calls):
        raise ValueError("zero or exactly the remaining plan/check calls required")
    pin = fingerprint()
    freeze = seal({"profile": witness.PROFILE, "requests": requests, "implementation": pin,
        "model": author.MODEL, "modelConfig": REVIEW_CONFIG, "maxNewModelCalls": max_model_calls,
        "retainedPlans": retained,
        "sourcePlannerSeesCandidate": False, "knownDevelopmentOnly": True, "semanticGold": False,
        "sourceScriptCalls": 0, "businessReadCalls": 0, "effectCalls": 0, "authorityGranted": False})
    write_artifacts(output / "freeze", {"manifest.json": freeze})
    with tarfile.open(output / "freeze/source-snapshot.tar.gz", "x:gz") as archive:
        for name in pin:
            archive.add(ROOT / name, arcname=name, recursive=False)
    rows = []
    for key, payload in payloads.items():
        if fingerprint() != pin:
            raise ValueError("execution implementation drift; stop without another call")
        source = witness.source_input(payload)
        plan_wire = wire_request(witness.PLAN_SYSTEM, witness.plan_input(source), witness.plan_schema(source))
        write_artifacts(output / key / "inputs", {"review-input.json": payload, "source-input.json": source,
                                                 "source-wire.json": plan_wire})
        if not max_model_calls:
            rows.append({"case": key, "status": "preflight_only", "sourceWireBudget": budget(plan_wire)})
            continue
        plan = ({"result": retained[key]["result"], "bound-opinion.json": retained[key]["boundPlan"]}
                if key in retained else call(output / key / "plan", plan_wire, lambda raw: witness.bind_plan(source, raw)))
        row = {"case": key, "planStatus": plan["result"]["status"], "semanticSuccess": None}
        row["planReusedWithoutCall"] = key in retained
        if fingerprint() != pin:
            raise ValueError("execution implementation drift after plan")
        if "bound-opinion.json" in plan:
            supplied = witness.check_input(payload, plan["bound-opinion.json"])
            wire = wire_request(witness.CHECK_SYSTEM, witness.check_guide(supplied), witness.check_schema(supplied))
            write_artifacts(output / key / "check-inputs", {"check-input.json": supplied, "check-wire.json": wire})
            checked = call(output / key / "check", wire, lambda raw: witness.bind_check(supplied, raw))
            row["checkStatus"] = checked["result"]["status"]
            if "bound-opinion.json" in checked:
                opinion = checked["bound-opinion.json"]
                row.update(findings=len(opinion["findings"]), bindingIssues=len(opinion["bindingIssues"]))
        rows.append(row)
        print(json.dumps(row), flush=True)
    if fingerprint() != pin:
        raise ValueError("execution implementation drift after checks")
    evidence = collect(output)
    write_artifacts(output / "evidence", {"report.json": evidence})
    report = seal({"profile": witness.PROFILE, "freezeDigest": freeze["reportDigest"], "rows": rows,
        "implementationUnchanged": True, "evidenceDigest": evidence["reportDigest"],
        **{k: evidence[k] for k in ("actualChatAttempts", "inputTokens", "outputTokens",
            "callsWithUnknownInputUsage", "callsWithUnknownOutputUsage", "p50CallLatencyMs", "p95CallLatencyMs")},
        "retainedPriorChatAttempts": len(retained),
        "retainedPriorInputTokens": sum(r["result"].get("inputTokens") or 0 for r in retained.values()),
        "retainedPriorOutputTokens": sum(r["result"].get("outputTokens") or 0 for r in retained.values()),
        "retainedPriorCallsWithUnknownInputUsage": sum(r["result"].get("inputTokens") is None for r in retained.values()),
        "retainedPriorCallsWithUnknownOutputUsage": sum(r["result"].get("outputTokens") is None for r in retained.values()),
        "candidateChanges": 0, "historicalGradesChanged": False, "semanticSuccess": None,
        "completeAnswerApproved": False, "stageExitMet": False})
    write_artifacts(output / "summary", {"report.json": report})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--max-model-calls", type=int, default=0)
    parser.add_argument("--reuse-plans", help="Explicit immutable prior run; only exact successful plan receipts are reusable")
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output, max_model_calls=args.max_model_calls, reuse_plans=args.reuse_plans), ensure_ascii=False))


if __name__ == "__main__":
    main()
