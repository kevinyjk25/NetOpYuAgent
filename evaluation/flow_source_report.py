"""Replay the two C3c development protocols; no model or provider calls."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.flow_grounded_translation import CitedDraft, assess_cited, load_cited, project
from evaluation.flow_source_selection import SelectedDraft, expand, load_selected
from evaluation.flow_translation import FlowSources, _write
from network_runtime.contracts import sha256_json
from evaluation.read_l05_review import ReadL05Review

ROOT = Path(__file__).resolve().parents[1]


def build_report(cited_root: Path, selected_root: Path, review_root: Path) -> dict:
    experiments, reviews = [], []
    frozen_sources = None
    for label, root in (("cited", cited_root), ("selected", selected_root)):
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["manifestDigest"] != sha256_json({k: v for k, v in manifest.items() if k != "manifestDigest"}):
            raise ValueError("pilot manifest digest mismatch")
        expected_ids = {"direct-read", "inverted-branch", "missing-approval-write", "unavailable-script-prerequisite"}
        source_map = {case["id"]: case["sources"] for case in manifest["cases"]}
        if len(manifest["cases"]) != 4 or set(source_map) != expected_ids or manifest["protocol"] != f"source-{label}-flow/v1":
            raise ValueError("report requires the declared four-case protocol pair")
        if frozen_sources is not None and source_map != frozen_sources:
            raise ValueError("protocol comparison requires identical source contexts")
        frozen_sources = source_map
        if any("sha256:" + hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != digest
               for p, digest in manifest["implementation"].items()):
            raise ValueError("pilot implementation drift")
        rows = []
        for case in manifest["cases"]:
            folder = root / case["id"]
            sources = FlowSources.model_validate_json((folder / "sources.json").read_text())
            request = json.loads((folder / "request.json").read_text())
            if sources != FlowSources.model_validate(case["sources"]) or request["model"] != manifest["model"] or sha256_json(request["wireRequest"]) != case["wireDigest"]:
                raise ValueError("actual source/request/model differs from frozen pilot")
            response = json.loads((folder / "response.json").read_text())
            status = json.loads((folder / "status.json").read_text())
            if response["httpStatus"] != 200:
                raise ValueError("this evidence report expects preserved completed model responses")
            reply = json.loads(response["body"])
            proposal = CitedDraft.model_validate_json(reply["message"]["content"]) if label == "cited" else expand(
                sources, SelectedDraft.model_validate_json(reply["message"]["content"]))
            if proposal.model_dump(mode="json") != json.loads((folder / "cited-proposal.json").read_text()):
                raise ValueError("saved proposal differs from raw response")
            try:
                project(sources, proposal)
            except ValueError as error:
                if status["status"] != "blocked" or status.get("error") != str(error):
                    raise ValueError("recorded qualification failure differs from replay") from error
            else:
                if status["status"] != "awaiting_source_review":
                    raise ValueError("recorded qualification status differs from replay")
                loaded = load_cited(folder) if label == "cited" else load_selected(folder)
                name = "cited-script" if label == "cited" else "selected-" + case["id"]
                review = ReadL05Review.model_validate_json((review_root / (name + ".json")).read_text())
                assessed = assess_cited(*loaded, review)
                if assessed != json.loads((review_root / (name + "-report.json")).read_text()):
                    raise ValueError("source assessment replay differs")
                reviews.append({"protocol": label, "case": case["id"], "status": assessed["status"],
                    "inputDigest": assessed["inputDigest"], "reviewDigest": assessed["reviewDigest"],
                    "reportDigest": assessed["reportDigest"], "verdictCounts": assessed["assessment"]["verdictCounts"],
                    "issues": assessed["issues"], "reviewerKind": review.reviewer_kind,
                    "findings": [{"pointer": row["l05Pointer"], "verdict": row["verdict"], "explanation": row["rationale"]}
                                 for row in assessed["assessment"]["rows"] if row["verdict"] != "supported"]})
            rows.append({"case": case["id"], "status": status, "latencyMs": response["latencyMs"],
                "inputTokens": reply.get("prompt_eval_count"), "outputTokens": reply.get("eval_count"),
                "files": {p.name: "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.json"))}})
        experiments.append({"protocol": manifest["protocol"], "manifestDigest": manifest["manifestDigest"], "model": manifest["model"],
            "rows": rows, "qualified": sum(row["status"]["status"] == "awaiting_source_review" for row in rows)})
    body = {"evidenceRole": "known_four_flow_protocol_development_not_holdout", "uniqueFlowCount": 4,
        "publicSkillCount": 0, "experiments": experiments, "reviews": reviews,
        "reviewSupportedInactiveFlows": sum(row["status"] == "review_supported_inactive_flow" for row in reviews),
        "runtimeExecutions": 0, "thirdPartyScriptExecutions": 0, "writes": 0, "independentHumanEvidence": False,
        "boundary": "Two different protocols on four known developer-selected cases, not accuracy or causal performance estimates. Citation/structure and semantic review remain separate. Same assistant authored and reviewed sources. Timings exclude preflight, source review and tests. Cited protocol is diagnostic history, not the default authoring path."}
    return {**body, "reportDigest": sha256_json(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cited_root", type=Path)
    parser.add_argument("selected_root", type=Path)
    parser.add_argument("review_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _write(args.output, build_report(args.cited_root, args.selected_root, args.review_root))


if __name__ == "__main__":
    main()
