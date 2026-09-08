"""Task-scoped source/host dossiers, not model judgments or Runtime admission.

Review decisions remain explicitly developer-authored. This module verifies
bindings and exposes missing context; it never infers semantic truth from a hash.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from evaluation.structured_binding_probe import read_json, write_artifacts
from evaluation.structured_flow_tree import SourceSpan
from evaluation.translation_intake import review_pages, validate_bundle
from network_runtime.contracts import sha256_json
from network_runtime.l0.models import StrictModel
from network_runtime.l0.structured_schema import DataBindingError, checked_schema, schema_location, schema_types, snapshot_json


class ReadRange(StrictModel):
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)


class DocumentReview(StrictModel):
    path: str
    ranges: tuple[ReadRange, ...] = Field(max_length=64)
    rationale: str = Field(min_length=12, max_length=1800)


class HostNeed(StrictModel):
    id: str
    kind: Literal["tool", "context", "policy"] = "tool"
    operation: str = Field(min_length=3)
    tool_name: str | None = None
    rationale: str = Field(min_length=12, max_length=1800)


class Obligation(StrictModel):
    id: str
    statement: str = Field(min_length=8, max_length=1800)
    disposition: Literal["candidate_read", "candidate_effect", "l1_reasoning", "constraint",
                         "host_precondition", "outside_task", "conditional", "unresolved"]
    citations: tuple[SourceSpan, ...] = Field(min_length=1, max_length=8)
    rationale: str = Field(min_length=12, max_length=1800)
    host_needs: tuple[str, ...] = Field(default=(), max_length=16)


class Finding(StrictModel):
    id: str
    category: Literal["source_tension", "missing_context", "unsupported_representation", "reference_gap"]
    explanation: str = Field(min_length=12, max_length=1800)
    citations: tuple[SourceSpan, ...] = Field(min_length=1, max_length=8)
    next_action: str = Field(min_length=12, max_length=1800)


class TaskAlignment(StrictModel):
    api_version: Literal["netopyu.io/task-alignment/v1"]
    bundle_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    task_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    # This is an evaluation request, never a new real-world user authorization.
    task: str = Field(min_length=12, max_length=4000)
    task_origin: Literal["developer_authored_evaluation_request"]
    task_basis: tuple[SourceSpan, ...] = Field(min_length=1, max_length=8)
    documents: tuple[DocumentReview, ...] = Field(min_length=1, max_length=128)
    obligations: tuple[Obligation, ...] = Field(min_length=1, max_length=128)
    host_needs: tuple[HostNeed, ...] = Field(min_length=1, max_length=32)
    findings: tuple[Finding, ...] = Field(max_length=64)
    # Human-supplied scope decisions only; targets are inert strings, never opened.
    dependencies: tuple["Dependency", ...] = Field(default=(), max_length=128)


class Dependency(StrictModel):
    target: str
    disposition: Literal["required_for_task", "outside_task", "template_example", "unresolved"]
    citation: SourceSpan
    rationale: str = Field(min_length=12, max_length=1800)


TaskAlignment.model_rebuild()


def _unique(items, key, label):
    result = {getattr(item, key): item for item in items}
    if len(result) != len(items) or any(not str(value).strip() for value in result):
        raise ValueError(f"duplicate or blank {label}")
    return result


def build_dossier(bundle: dict, alignment: TaskAlignment, catalog: dict | None = None) -> dict[str, Any]:
    bundle = snapshot_json(bundle)
    catalog = snapshot_json(catalog) if catalog is not None else None
    validate_bundle(bundle)
    alignment = TaskAlignment.model_validate(alignment.model_dump(mode="json"))
    if alignment.bundle_digest != bundle["bundleDigest"]:
        raise ValueError("task alignment source bundle drift")
    docs = {d["path"]: d for d in bundle["documents"]}
    reviews = _unique(alignment.documents, "path", "document review")
    if set(reviews) != set(docs):
        raise ValueError("every retained document needs an explicit review disposition")
    review_rows = []
    for path, review in reviews.items():
        text, end, read_characters = docs[path].get("content"), 0, 0
        for span in review.ranges:
            if (docs[path]["representation"] != "inert_utf8_text" or not isinstance(text, str)
                    or span.start < end or not span.start < span.end <= len(text)):
                raise ValueError("invalid, overlapping or unordered document review ranges")
            read_characters += span.end - span.start
            end = span.end
        review_rows.append({**review.model_dump(mode="json"), "sourceDigest": docs[path]["sha256"],
                            "reviewedCharacters": read_characters, "totalCharacters": len(text) if isinstance(text, str) else 0,
                            "fullTextReviewedDeclared": bool(text) and read_characters == len(text)})
    if not reviews[bundle["entryPath"]].ranges:
        raise ValueError("task review must include the Skill entry")

    def citation(span):
        doc = docs.get(span.path)
        if (not doc or doc["representation"] != "inert_utf8_text"
                or doc["content"][span.start:span.end] != span.quote or span.end - span.start != len(span.quote)):
            raise ValueError("citation must match the exact retained source")
        if not any(r.start <= span.start < span.end <= r.end for r in reviews[span.path].ranges):
            raise ValueError("citation is outside declared reviewed source ranges")
        return {**span.model_dump(mode="json"), "sourceDigest": doc["sha256"],
                "line": doc["content"].count("\n", 0, span.start) + 1, "semanticEntailmentProven": False}

    task_basis = [citation(c) for c in alignment.task_basis]
    needs = _unique(alignment.host_needs, "id", "host need")
    _unique(alignment.obligations, "id", "obligation")
    _unique(alignment.findings, "id", "finding")
    obligations = []
    for obligation in alignment.obligations:
        if len(set(obligation.host_needs)) != len(obligation.host_needs) or set(obligation.host_needs) - needs.keys():
            raise ValueError("obligation has duplicate or unknown host requirements")
        if obligation.disposition in {"candidate_read", "candidate_effect"} and not obligation.host_needs:
            raise ValueError("candidate operations must declare their host requirements")
        obligations.append({**obligation.model_dump(mode="json"), "citations": [citation(c) for c in obligation.citations]})
    findings = []
    for finding in alignment.findings:
        if finding.category == "source_tension" and len({(c.path, c.start, c.end) for c in finding.citations}) < 2:
            raise ValueError("source tension needs at least two distinct source passages")
        findings.append({**finding.model_dump(mode="json"), "citations": [citation(c) for c in finding.citations], "resolved": False})
    dependencies = [{**d.model_dump(mode="json"), "citation": citation(d.citation),
                     "retainedTextPresent": d.target in docs and docs[d.target]["representation"] == "inert_utf8_text"}
                    for d in alignment.dependencies]
    host_rows, tools = [], {}
    if catalog is not None:
        declarations = catalog.get("tools")
        if not isinstance(declarations, list) or not 1 <= len(declarations) <= 128:
            raise ValueError("catalog requires a bounded nonempty tools list")
        for tool in declarations:
            if (not isinstance(tool, dict) or not isinstance(tool.get("name"), str)
                    or not tool["name"].strip() or tool["name"] in tools):
                raise ValueError("host tools require unique nonblank names")
            tools[tool["name"]] = tool
    for need in needs.values():
        if need.kind != "tool":
            if need.tool_name is not None:
                raise ValueError("host context/policy requirements cannot be satisfied by a tool name")
            host_rows.append({**need.model_dump(mode="json"), "toolDigest": None, "shapeReadyOnly": None,
                              "diagnostics": [{"code": "host_" + need.kind + "_not_bound", "pointer": "",
                                               "detail": "Requires separate host review, not an invented API or a tools/list entry."}],
                              "sourceApiMappingVerified": False, "hostAuthenticityVerified": False})
            continue
        tool = tools.get(need.tool_name)
        gaps = []
        if tool is None:
            gaps.append({"code": "host_binding_missing", "pointer": "", "detail": "No exact named host declaration supplied."})
        else:
            for role in ("inputSchema", "outputSchema"):
                try:
                    schema = checked_schema(tool.get(role))
                    if schema_types(schema_location(schema, "")[0]) != {"object"}:
                        gaps.append({"code": "host_root_not_object", "pointer": "", "role": role,
                                     "detail": "The current structured read gateway requires object roots."})
                except DataBindingError as error:
                    gaps.append({**error.as_dict(), "role": role})
        host_rows.append({**need.model_dump(mode="json"), "toolDigest": sha256_json(tool) if tool else None,
                          "shapeReadyOnly": not gaps, "diagnostics": gaps,
                          "sourceApiMappingVerified": False, "hostAuthenticityVerified": False})
    pages = review_pages(bundle)
    # This future author input deliberately has no developer obligations/verdicts.
    model_input = {"task": alignment.task, "taskOrigin": alignment.task_origin,
                   "sourceBundleDigest": bundle["bundleDigest"], "sourcePages": pages,
                   "declaredHostCatalog": catalog, "thirdPartyContentExecutable": False,
                   "goldIncluded": False, "reviewDecisionsIncluded": False}
    model_input["inputDigest"] = sha256_json(model_input)
    body = {"apiVersion": "netopyu.io/task-dossier/v1", "candidateId": bundle["candidateId"],
            "taskId": alignment.task_id, "task": alignment.task, "taskOrigin": alignment.task_origin,
            "bundleDigest": bundle["bundleDigest"], "alignmentDigest": sha256_json(alignment.model_dump(mode="json")),
            "catalogDigest": sha256_json(catalog) if catalog is not None else None,
            "modelInputDigest": model_input["inputDigest"], "taskBasis": task_basis,
            "documentReviews": review_rows, "obligations": obligations, "findings": findings,
            "hostRequirements": host_rows, "dependencies": dependencies,
            "lexicalReferenceCandidates": bundle["references"],
            "dispositionCounts": dict(Counter(o.disposition for o in alignment.obligations)),
            "missingHostRequirements": sum(not r["shapeReadyOnly"] for r in host_rows),
            "status": "review_dossier_bound_not_translation_qualified",
            "reviewerKind": "development_assistant_not_independent_gold",
            "reviewRangeBindingVerified": True, "semanticEntailmentProven": False,
            "wholeSourceCoverageProven": False, "taskReferenceClosureProven": False,
            "wholeSkillTranslationProven": False, "runtimeAuthorityGranted": False,
            "modelCalls": 0, "providerCalls": 0, "translationMetrics": None,
            "claimBoundary": "Declarations/citations are review input, not semantic truth, host authenticity or execution approval."}
    report = {**body, "reportDigest": sha256_json(body)}
    return {"alignment.json": alignment.model_dump(mode="json"), "source-bundle.json": bundle,
            "host-catalog.json": catalog, "model-input.json": model_input, "report.json": report}


def prepare(bundle_path, alignment_path, output, *, catalog_path=None):
    if Path(output).exists():
        raise FileExistsError("output exists; preserve previous alignment evidence")
    files = build_dossier(read_json(bundle_path), TaskAlignment.model_validate(read_json(alignment_path)),
                          read_json(catalog_path) if catalog_path else None)
    write_artifacts(output, files)
    return files["report.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("alignment")
    parser.add_argument("--host-catalog")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = prepare(args.bundle, args.alignment, args.output, catalog_path=args.host_catalog)
    print(result["taskId"], result["status"], "missing-hosts:", result["missingHostRequirements"])


if __name__ == "__main__":
    main()
