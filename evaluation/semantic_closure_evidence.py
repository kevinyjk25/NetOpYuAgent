"""Audit frozen local evidence and total cost, without regrading old semantics."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

from evaluation.flow_tree_authoring import verify_receipt
from evaluation.structured_authoring import seal
from evaluation.structured_binding_probe import write_artifacts
from network_runtime.contracts import sha256_json
from network_runtime.l0.read_contracts import _source_object

MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


def read_json(path):
    """Bounded audit metadata reader, not a change to Runtime input limits.

    Frozen requests may duplicate large immutable graph literals. They remain
    fully checked; neither omit oversized manifests nor extract source archives.
    """
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_EVIDENCE_BYTES + 1)
    if len(raw) > MAX_EVIDENCE_BYTES:
        raise ValueError("evidence JSON exceeds audit byte budget")
    return _source_object(raw.decode("utf-8"))


def percentile(values, q):
    if not values:
        return None
    rows = sorted(values)
    position = (len(rows) - 1) * q
    left, right = int(position), min(int(position) + 1, len(rows) - 1)
    return rows[left] + (rows[right] - rows[left]) * (position - left)


def reconstruct_duty_checkpoint(path, root, value):
    """Recognize one exact historical nested-digest collision, never arbitrary drift.

The original file stays invalid and untouched. The linked case report must be
independently sealed and reproduce the historical hash computation exactly.
"""
    relative = path.relative_to(root)
    required = {"case", "status", "reportDigest"}
    optional = {"allScheduledNodesBound", "failedNodeCount"}
    if (len(relative.parts) != 3 or relative.parts[1:] != ("checkpoint", "report.json")
            or set(value) not in (required, required | optional)
            or value["case"] != relative.parts[0] or value["status"] != "bounded_diagnostic_completed"):
        raise ValueError("not a recognized duty checkpoint collision")
    linked = read_json(root / relative.parts[0] / "summary/report.json")
    if linked != seal({k: v for k, v in linked.items() if k != "reportDigest"}):
        raise ValueError("linked case report drift")
    if optional <= set(value) and (value["allScheduledNodesBound"] != linked["allScheduledNodesBound"]
                                  or value["failedNodeCount"] != len(linked["failures"])):
        raise ValueError("case checkpoint state drift")
    body = {k: v for k, v in value.items() if k != "reportDigest"}
    if seal({**body, "reportDigest": linked["reportDigest"]}) != value:
        raise ValueError("unrecognized checkpoint digest computation")
    return {"path": str(relative), "originalFileDigest": sha256_json(value), "originalSelfSealValid": False,
            "cause": "linked_case_digest_overwritten_by_self_digest",
            "reconstructed": seal({**body, "caseReportDigest": linked["reportDigest"]}),
            "originalFileChanged": False, "historicalGradeChanged": False}


def collect(root, *, reconstruct_duty_checkpoints=False):
    root = Path(root).resolve()
    calls, bound, archives, reports, reconstructions = [], {}, [], [], []
    for path in sorted(root.rglob("request.json")):
        if not (path.parent / "response.json").exists():
            # Only real model requests carry wireRequest. Other construction
            # requests have no model side effect and are not counted as calls.
            if "wireRequest" in read_json(path):
                raise ValueError(f"model attempt still incomplete: {path.relative_to(root)}")
            continue
        verify_receipt(path.parent)
        result = read_json(path.parent / "result.json")
        raw = read_json(path)
        calls.append({"path": str(path.parent.relative_to(root)), "model": raw.get("model"),
                      "thinkingRequested": raw.get("wireRequest", {}).get("think"), **result})
        for name, digest in read_json(path.parent / "receipt.json").items():
            bound[str((path.parent / name).relative_to(root))] = digest
    for path in sorted(root.rglob("source-snapshot.tar.gz")):
        candidates = [read_json(p) for p in path.parent.glob("*.json")]
        owners = [r for r in candidates if isinstance(r.get("implementation"), dict)]
        if len(owners) != 1:
            raise ValueError(f"archive lacks one implementation manifest: {path.relative_to(root)}")
        pinned = owners[0]["implementation"]
        with tarfile.open(path, "r:gz") as archive:
            if set(archive.getnames()) != set(pinned):
                raise ValueError("source archive membership differs from freeze")
            for name, expected in pinned.items():
                member = archive.getmember(name)
                if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise ValueError("archive contains a nonregular or unconfined source")
                with archive.extractfile(member) as source:
                    actual = "sha256:" + hashlib.sha256(source.read()).hexdigest()
                if actual != expected:
                    raise ValueError(f"frozen source digest drift: {path.relative_to(root)}:{name}")
        archives.append({"path": str(path.relative_to(root)), "sourceFiles": len(pinned),
                         "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()})
    for path in sorted(root.rglob("report.json")):
        value = read_json(path)
        if value != seal({k: v for k, v in value.items() if k != "reportDigest"}):
            if not reconstruct_duty_checkpoints:
                raise ValueError(f"report digest drift: {path.relative_to(root)}")
            reconstructions.append(reconstruct_duty_checkpoint(path, root, value))
        reports.append({"path": str(path.relative_to(root)), "reportDigest": value["reportDigest"],
                        "status": value.get("status", value.get("execution", {}).get("status")),
                        "semanticSuccess": value.get("semanticSuccess")})
    latencies = [c["latencyMs"] for c in calls if isinstance(c.get("latencyMs"), (int, float))]
    return seal({"status": "evidence_collected_with_disclosed_checkpoint_reconstruction" if reconstructions else "evidence_collected_not_semantic_acceptance", "root": str(root),
        **({"allOriginalReportSealsValid": False, "checkpointReconstructions": reconstructions} if reconstructions else {}),
        "actualChatAttempts": len(calls), "modelCallStatusCounts": dict(Counter(c.get("status", "unknown") for c in calls)),
        "inputTokens": sum(c.get("inputTokens") or 0 for c in calls),
        "outputTokens": sum(c.get("outputTokens") or 0 for c in calls),
        "callsWithUnknownInputUsage": sum(c.get("inputTokens") is None for c in calls),
        "callsWithUnknownOutputUsage": sum(c.get("outputTokens") is None for c in calls),
        "callsRequestingNativeThinking": sum(c.get("thinkingRequested") is True for c in calls),
        "tokenAccountingBoundary": "Provider-reported counters only; separate native reasoning-token attribution may be unavailable.",
        "p50CallLatencyMs": percentile(latencies, .5), "p95CallLatencyMs": percentile(latencies, .95),
        "latencyMeaning": "All retained model calls across development versions, not task latency, SLO or causal A/B gain.",
        "calls": calls, "archives": archives, "boundArtifactDigests": bound, "reports": reports,
        "semanticSuccessRate": None, "largeRuntimeABUnlocked": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("output")
    args = parser.parse_args()
    result = collect(args.root)
    write_artifacts(args.output, {"report.json": result})
    print(json.dumps({k: result[k] for k in ("reportDigest", "actualChatAttempts", "inputTokens", "outputTokens")}), flush=True)


if __name__ == "__main__":
    main()
